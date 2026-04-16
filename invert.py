import pygame
import numpy as np
import sys
from collections import deque
from active_inference import ActiveInferenceController

# --- Configuration & Window ---
W, H   = 1000, 700
FPS    = 60
# --- Physics Constants ---
DT         = 1.0 / FPS
G          = 9.81
L          = 1.0
M          = 1.0
DAMPING    = 0.25
MAX_TORQUE = 15.0
# --- Layout Geometry ---
PIVOT_X    = W // 2
PIVOT_Y    = H // 2 + 100
ARM_PX     = 220
BASE_W     = 80
BASE_H     = 24
TRACK_Y    = PIVOT_Y + BASE_H // 2 + 4
# --- Synthwave Palette ---
BG         = (8, 10, 15)
GRID       = (20, 25, 35)
PANEL_BG   = (15, 18, 25)
WHITE      = (240, 245, 255)
DIM        = (70, 85, 110)
TEAL       = (0, 230, 190)
PINK       = (255, 40, 120)
ORANGE     = (255, 140, 20)
BLUE_GLOW  = (20, 100, 255)
VIOLET     = (180, 80, 255)
CYAN       = (0, 200, 255)

class Pendulum:
    """
    Inverted Pendulum.
    theta = 0  -> Upright
    theta = ±π -> Hanging down
    """
    def __init__(self, theta0=0.15):
        self.theta    = float(theta0)
        self.thetadot = 0.0

    def step(self, torque):
        torque = np.clip(torque, -MAX_TORQUE, MAX_TORQUE)
        alpha  = (G / L)*np.sin(self.theta) + torque/(M*L**2) - DAMPING*self.thetadot
        self.thetadot += alpha * DT
        self.theta    += self.thetadot * DT
        self.theta     = (self.theta + np.pi) % (2 * np.pi) - np.pi

    def get_energy(self):
        ke = 0.5 * (M * L**2) * self.thetadot**2
        pe = M * G * L * np.cos(self.theta)
        return ke + pe

def glow_circle(surf, col, center, r, layers=4):
    for i in range(layers, 0, -1):
        alpha = int(60 * (1 - i / (layers + 1)))
        s = pygame.Surface((r * 2 * i + 4, r * 2 * i + 4), pygame.SRCALPHA)
        pygame.draw.circle(s, (*col, alpha), (r * i + 2, r * i + 2), r * i)
        surf.blit(s, (center[0] - r * i - 2, center[1] - r * i - 2))
    pygame.draw.circle(surf, col, center, r)

def lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def draw_hud_panel(surf, rect, col, radius=8, alpha=230):
    s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
    pygame.draw.rect(s, (*col, alpha), (0, 0, rect[2], rect[3]), border_radius=radius)
    pygame.draw.rect(s, (*DIM, 80), (0, 0, rect[2], rect[3]), 1, border_radius=radius)
    surf.blit(s, (rect[0], rect[1]))

def draw_grid(surf):
    for x in range(0, W, 40):
        pygame.draw.line(surf, GRID, (x, 0), (x, H), 1)
    for y in range(0, H, 40):
        pygame.draw.line(surf, GRID, (0, y), (W, y), 1)

def draw_sparkline(surf, data, rect, col, lo, hi, label, font):
    x, y, w, h = rect
    draw_hud_panel(surf, rect, PANEL_BG, radius=4, alpha=180)
    zero_t  = np.clip(-lo / (hi - lo + 1e-9), 0, 1)
    zero_py = y + h - int(zero_t * h)
    pygame.draw.line(surf, DIM, (x, zero_py), (x + w, zero_py), 1)
    pts = []
    n = len(data)
    for i, val in enumerate(data):
        px = x + int(i / max(n - 1, 1) * w)
        t  = (val - lo) / max(hi - lo, 1e-9)
        py = y + h - int(np.clip(t, 0, 1) * h)
        pts.append((px, py))
    if len(pts) > 1:
        pygame.draw.lines(surf, col, False, pts, 2)
    surf.blit(font.render(label, True, WHITE), (x + 6, y + 4))

def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Inverted Pendulum")
    clock  = pygame.time.Clock()
    font_lg = pygame.font.SysFont("consolas", 22, bold=True)
    font_md = pygame.font.SysFont("consolas", 16)
    font_sm = pygame.font.SysFont("consolas", 12)

    pend = Pendulum(theta0=0.2)
    ai   = ActiveInferenceController(dt=DT)
    ai.reset(theta0=0.2)

    MODES    = ["manual", "ai"]
    mode_idx = 0
    mode     = MODES[mode_idx]

    step   = 0
    paused = False

    override_active = False

    HIST_LEN    = 200
    hist_theta  = deque([0.0] * HIST_LEN, maxlen=HIST_LEN)
    hist_torque = deque([0.0] * HIST_LEN, maxlen=HIST_LEN)
    hist_energy = deque([0.0] * HIST_LEN, maxlen=HIST_LEN)

    hist_free_energy = deque([0.0] * HIST_LEN, maxlen=HIST_LEN)
    hist_pred_err    = deque([0.0] * HIST_LEN, maxlen=HIST_LEN)
    hist_belief_err  = deque([0.0] * HIST_LEN, maxlen=HIST_LEN)

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                if ev.key == pygame.K_SPACE:
                    paused = not paused
                if ev.key == pygame.K_r:
                    t0 = np.random.uniform(-0.5, 0.5)
                    pend = Pendulum(theta0=t0)
                    ai.reset(theta0=t0)
                    for h in [hist_theta, hist_torque, hist_energy,
                            hist_free_energy, hist_pred_err, hist_belief_err]:
                        h.clear(); h.extend([0.0] * HIST_LEN)
                    step = 0
                if ev.key == pygame.K_TAB:
                    mode_idx = (mode_idx + 1) % len(MODES)
                    mode = MODES[mode_idx]
                if ev.key == pygame.K_LEFTBRACKET:
                    ai.pi_s = np.clip(ai.pi_s - 2.0, 1.0, 50.0)
                if ev.key == pygame.K_RIGHTBRACKET:
                    ai.pi_s = np.clip(ai.pi_s + 2.0, 1.0, 50.0)

        torque          = 0.0
        override_active = False

        if not paused:
            keys = pygame.key.get_pressed()

            if mode == "manual":
                if keys[pygame.K_LEFT]:
                    torque = -MAX_TORQUE
                if keys[pygame.K_RIGHT]:
                    torque = MAX_TORQUE

            elif mode == "ai":
                manual_torque = 0.0
                if keys[pygame.K_LEFT]:
                    manual_torque = -MAX_TORQUE
                    override_active = True
                if keys[pygame.K_RIGHT]:
                    manual_torque = MAX_TORQUE
                    override_active = True

                ai_torque = ai.action(pend.theta, pend.thetadot)

                if override_active:
                    torque = manual_torque * 0.85 + ai_torque * 0.15
                else:
                    torque = ai_torque

            pend.step(torque)
            hist_theta.append(pend.theta)
            hist_torque.append(torque)
            hist_energy.append(pend.get_energy())

            diag = ai.get_diagnostics()
            hist_free_energy.append(diag["free_energy"])
            hist_pred_err.append(float(np.linalg.norm(diag["eps_s"])))
            hist_belief_err.append(abs(diag["belief_angle"] - pend.theta))
            step += 1

        pivot    = (PIVOT_X, PIVOT_Y)
        tip_x    = PIVOT_X + int(ARM_PX * np.sin(pend.theta))
        tip_y    = PIVOT_Y - int(ARM_PX * np.cos(pend.theta))
        tip      = (tip_x, tip_y)
        ang_frac = min(abs(pend.theta) / (np.pi / 2), 1.0)
        arm_col  = lerp_color(TEAL, PINK, ang_frac)

        screen.fill(BG)
        draw_grid(screen)
        pygame.draw.line(screen, DIM, (0, TRACK_Y), (W, TRACK_Y), 2)

        base_rect = (PIVOT_X - BASE_W//2, PIVOT_Y - BASE_H//2, BASE_W, BASE_H)
        if mode == "ai" and override_active:
            mount_col = lerp_color(CYAN, ORANGE, 0.7)
        elif mode == "ai":
            mount_col = CYAN
        else:
            mount_col = TEAL

        draw_hud_panel(screen, base_rect, PANEL_BG, radius=4, alpha=255)
        pygame.draw.rect(screen, mount_col, base_rect, 1, border_radius=4)

        ghost_tip = (PIVOT_X, PIVOT_Y - ARM_PX)
        pygame.draw.line(screen, (*TEAL, 50), pivot, ghost_tip, 2)
        pygame.draw.line(screen, arm_col, pivot, tip, 6)
        pygame.draw.circle(screen, WHITE, pivot, 8)
        pygame.draw.circle(screen, BG,    pivot, 4)
        glow_circle(screen, arm_col, tip, 18, layers=4)
        pygame.draw.circle(screen, WHITE, tip, 6)

        if abs(torque) > 0.1:
            if mode == "ai" and override_active:
                arrow_col = ORANGE
            elif mode == "ai":
                arrow_col = CYAN
            else:
                arrow_col = ORANGE
            arrow_dir = -1 if torque < 0 else 1
            arrow_x   = PIVOT_X + arrow_dir * 40
            pygame.draw.line(screen, arrow_col, (PIVOT_X, PIVOT_Y), (arrow_x, PIVOT_Y), 4)
            pygame.draw.polygon(screen, arrow_col, [
                (arrow_x + arrow_dir * 8, PIVOT_Y),
                (arrow_x, PIVOT_Y - 6),
                (arrow_x, PIVOT_Y + 6),
            ])

        lh_base = 270 if mode != "ai" else 345
        LX, LY, LW, LH = 20, 20, 230, lh_base
        draw_hud_panel(screen, (LX, LY, LW, LH), PANEL_BG)
        screen.blit(font_lg.render("SYS.TELEMETRY", True, TEAL), (LX + 15, LY + 15))
        pygame.draw.line(screen, TEAL, (LX + 15, LY + 45), (LX + LW - 15, LY + 45), 2)

        mode_labels = {"manual": "MANUAL DRIVE", "ai": "ACTIVE INFERENCE"}
        mode_cols   = {"manual": ORANGE, "ai": CYAN}
        mode_label  = mode_labels[mode]
        mode_col    = mode_cols[mode]

        if mode == "ai" and override_active:
            badge_text = "[ MANUAL OVERRIDE ]"
            badge_col  = ORANGE
        else:
            badge_text = f"[ {mode_label} ]"
            badge_col  = mode_col

        badge = font_sm.render(badge_text, True, badge_col)
        screen.blit(badge, (LX + LW//2 - badge.get_width()//2, LY + 50))

        stats = [
            ("ANGLE (θ)",  f"{np.degrees(pend.theta):+7.1f}°",  arm_col),
            ("VELOCITY",   f"{pend.thetadot:+7.2f} r/s",        DIM),
            ("TORQUE (τ)", f"{torque:+7.2f} Nm",                 mode_col if not override_active else ORANGE),
            ("ENERGY",     f"{pend.get_energy():+7.2f} J",       PINK),
            ("SYS CLOCK",  f"{step:7d}",                         WHITE),
        ]
        if mode == "ai":
            d = ai.get_diagnostics()
            stats += [
                ("FREE ENRGY", f"{d['free_energy']:+7.2f}",      CYAN),
                ("Πₛ (sens)",  f"{d['pi_s'][0]:7.1f}",           CYAN),
            ]

        for i, (lbl, val, col) in enumerate(stats):
            yo = LY + 75 + i * 36
            screen.blit(font_sm.render(lbl, True, DIM), (LX + 15, yo))
            vs = font_md.render(val, True, col)
            screen.blit(vs, (LX + LW - vs.get_width() - 15, yo))

        RX, RY = W - 270, 20
        if mode == "ai":
            rh_h = 370
            draw_hud_panel(screen, (RX - 10, RY, 260, rh_h), PANEL_BG)
            draw_sparkline(screen, list(hist_theta),
                        (RX, RY + 15, 240, 60),
                        TEAL, -np.pi, np.pi, "Angle theta (rad)", font_sm)
            draw_sparkline(screen, list(hist_torque),
                        (RX, RY + 90, 240, 60),
                        CYAN, -MAX_TORQUE, MAX_TORQUE, "Torque tau (Nm)", font_sm)
            max_fe = max(max(hist_free_energy), 1.0)
            draw_sparkline(screen, list(hist_free_energy),
                        (RX, RY + 165, 240, 60),
                        PINK, 0, max_fe, "Free Energy F", font_sm)
            max_pe = max(max(hist_pred_err), 0.1)
            draw_sparkline(screen, list(hist_pred_err),
                        (RX, RY + 240, 240, 60),
                        ORANGE, 0, max_pe, "Prediction Error |eps_s|", font_sm)
            max_be = max(max(hist_belief_err), 0.1)
            draw_sparkline(screen, list(hist_belief_err),
                        (RX, RY + 315, 240, 40),
                        VIOLET, 0, max_be, "Belief Error |mu-theta|", font_sm)
        else:
            draw_hud_panel(screen, (RX - 10, RY, 260, 280), PANEL_BG)
            draw_sparkline(screen, list(hist_theta),
                        (RX, RY + 15, 240, 70),
                        TEAL, -np.pi, np.pi, "Angle theta (rad)", font_sm)
            draw_sparkline(screen, list(hist_torque),
                        (RX, RY + 100, 240, 70),
                        mode_col, -MAX_TORQUE, MAX_TORQUE, "Torque tau (Nm)", font_sm)
            max_e = max(hist_energy) if max(hist_energy) > 20 else 20
            draw_sparkline(screen, list(hist_energy),
                        (RX, RY + 185, 240, 70),
                        PINK, -15, max_e, "Total Energy (J)", font_sm)

        BY = H - 50
        draw_hud_panel(screen, (0, BY, W, 50), PANEL_BG, radius=0)
        if mode == "ai":
            controls = "[←/→] Override AI  [TAB] Mode  [SPACE] Pause  [R] Reset  [/] Pi_s  [ESC] Exit"
        else:
            controls = "[←/→] Torque  [TAB] Mode  [SPACE] Pause  [R] Reset  [ESC] Exit"
        cs = font_md.render(controls, True, DIM)
        screen.blit(cs, (W//2 - cs.get_width()//2, BY + 16))

        if paused:
            ov = pygame.Surface((W, H), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 150))
            screen.blit(ov, (0, 0))
            screen.blit(font_lg.render("SYSTEM PAUSED", True, TEAL),
                        (W//2 - 100, H//2 - 100))

        pygame.display.flip()
        clock.tick(FPS)

if __name__ == "__main__":
    main()
