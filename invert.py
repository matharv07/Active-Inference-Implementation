import pygame
import numpy as np
import sys
from collections import deque

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
VIOLET     = (180, 80, 255)   # PD controller colour

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
        alpha  = (G / L) * np.sin(self.theta) \
               + torque / (M * L * L) \
               - DAMPING * self.thetadot
        self.thetadot += alpha * DT
        self.theta    += self.thetadot * DT
        self.theta     = (self.theta + np.pi) % (2 * np.pi) - np.pi

    def get_energy(self):
        ke = 0.5 * (M * L**2) * self.thetadot**2
        pe = M * G * L * np.cos(self.theta)
        return ke + pe


# =========================================================================
#  PD Controller  (baseline — will be replaced by active inference)
# =========================================================================
class PDController:
    """
    Proportional-Derivative controller on theta.
    tau = -kp * theta  -  kd * thetadot

    Tuning intuition:
      kp  -> how hard it pulls toward upright (too high = oscillation)
      kd  -> how much it damps velocity       (too low = overshoot)

    This is the plug-in point. Active inference will expose the same
    interface: action(theta, thetadot) -> float torque
    """
    def __init__(self, kp=35.0, kd=7.0):
        self.kp = kp
        self.kd = kd

    def action(self, theta, thetadot):
        torque = -self.kp * theta - self.kd * thetadot
        return float(np.clip(torque, -MAX_TORQUE, MAX_TORQUE))


# =========================================================================
#  Drawing Utilities
# =========================================================================
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


# =========================================================================
#  Main Loop
# =========================================================================
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Inverted Pendulum")
    clock  = pygame.time.Clock()

    font_lg = pygame.font.SysFont("consolas", 22, bold=True)
    font_md = pygame.font.SysFont("consolas", 16)
    font_sm = pygame.font.SysFont("consolas", 12)

    pend = Pendulum(theta0=0.2)
    pd   = PDController()

    # Modes: "manual" | "pd"
    mode   = "manual"
    step   = 0
    paused = False

    HIST_LEN    = 200
    hist_theta  = deque([0.0] * HIST_LEN, maxlen=HIST_LEN)
    hist_torque = deque([0.0] * HIST_LEN, maxlen=HIST_LEN)
    hist_energy = deque([0.0] * HIST_LEN, maxlen=HIST_LEN)

    while True:
        # ── Events ────────────────────────────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                if ev.key == pygame.K_SPACE:
                    paused = not paused
                if ev.key == pygame.K_r:
                    pend = Pendulum(theta0=np.random.uniform(-0.5, 0.5))
                    for h in [hist_theta, hist_torque, hist_energy]:
                        h.clear(); h.extend([0.0] * HIST_LEN)
                    step = 0
                if ev.key == pygame.K_TAB:
                    mode = "pd" if mode == "manual" else "manual"

        # ── Torque ────────────────────────────────────────────────────────
        torque = 0.0
        if not paused:
            keys = pygame.key.get_pressed()

            if mode == "manual":
                if keys[pygame.K_LEFT]:
                    torque = -MAX_TORQUE
                if keys[pygame.K_RIGHT]:
                    torque = MAX_TORQUE

            elif mode == "pd":
                torque = pd.action(pend.theta, pend.thetadot)

            pend.step(torque)
            hist_theta.append(pend.theta)
            hist_torque.append(torque)
            hist_energy.append(pend.get_energy())
            step += 1

        # ── Geometry ──────────────────────────────────────────────────────
        pivot    = (PIVOT_X, PIVOT_Y)
        tip_x    = PIVOT_X + int(ARM_PX * np.sin(pend.theta))
        tip_y    = PIVOT_Y - int(ARM_PX * np.cos(pend.theta))
        tip      = (tip_x, tip_y)
        ang_frac = min(abs(pend.theta) / (np.pi / 2), 1.0)
        arm_col  = lerp_color(TEAL, PINK, ang_frac)

        # ── Render ────────────────────────────────────────────────────────
        screen.fill(BG)
        draw_grid(screen)

        pygame.draw.line(screen, DIM, (0, TRACK_Y), (W, TRACK_Y), 2)

        base_rect  = (PIVOT_X - BASE_W//2, PIVOT_Y - BASE_H//2, BASE_W, BASE_H)
        mount_col  = VIOLET if mode == "pd" else TEAL
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
            arrow_col = VIOLET if mode == "pd" else ORANGE
            arrow_dir = -1 if torque < 0 else 1
            arrow_x   = PIVOT_X + arrow_dir * 40
            pygame.draw.line(screen, arrow_col, (PIVOT_X, PIVOT_Y), (arrow_x, PIVOT_Y), 4)
            pygame.draw.polygon(screen, arrow_col, [
                (arrow_x + arrow_dir * 8, PIVOT_Y),
                (arrow_x, PIVOT_Y - 6),
                (arrow_x, PIVOT_Y + 6),
            ])

        # ── Left HUD ──────────────────────────────────────────────────────
        LX, LY, LW, LH = 20, 20, 230, 270
        draw_hud_panel(screen, (LX, LY, LW, LH), PANEL_BG)
        screen.blit(font_lg.render("SYS.TELEMETRY", True, TEAL), (LX + 15, LY + 15))
        pygame.draw.line(screen, TEAL, (LX + 15, LY + 45), (LX + LW - 15, LY + 45), 2)

        mode_label = "PD CONTROLLER" if mode == "pd" else "MANUAL DRIVE"
        mode_col   = VIOLET if mode == "pd" else ORANGE
        badge      = font_sm.render(f"[ {mode_label} ]", True, mode_col)
        screen.blit(badge, (LX + LW//2 - badge.get_width()//2, LY + 50))

        stats = [
            ("ANGLE (θ)",  f"{np.degrees(pend.theta):+7.1f}°",  arm_col),
            ("VELOCITY",   f"{pend.thetadot:+7.2f} r/s",        DIM),
            ("TORQUE (τ)", f"{torque:+7.2f} Nm",                 mode_col),
            ("ENERGY",     f"{pend.get_energy():+7.2f} J",       PINK),
            ("SYS CLOCK",  f"{step:7d}",                         WHITE),
        ]
        for i, (lbl, val, col) in enumerate(stats):
            yo = LY + 75 + i * 36
            screen.blit(font_sm.render(lbl, True, DIM), (LX + 15, yo))
            vs = font_md.render(val, True, col)
            screen.blit(vs, (LX + LW - vs.get_width() - 15, yo))

        # ── Right HUD (sparklines) ─────────────────────────────────────────
        RX, RY = W - 270, 20
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

        # ── Bottom bar ────────────────────────────────────────────────────
        BY = H - 50
        draw_hud_panel(screen, (0, BY, W, 50), PANEL_BG, radius=0)
        controls = "[←/→] Torque   [TAB] Toggle PD/Manual   [SPACE] Pause   [R] Reset   [ESC] Exit"
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
