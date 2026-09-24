import { useEffect, useRef, useState } from "react";

/**
 * How settled the model is, from how much it has been told.
 *
 * Derived rather than measured. The audit endpoint reports real interval
 * calibration, but it refits once per verdict and takes seconds — too much
 * for a figure. Home and Taste share this so the two pictures use one curve.
 */
export function settledness(verdicts) {
  return Math.min(0.95, 0.25 + verdicts / 400);
}

/**
 * How far along the path from cold start it has come, 0 to 1.
 *
 * Logarithmic because learning is: the first dozen verdicts move it further
 * than the next hundred.
 */
function travelled(verdicts) {
  return Math.min(1, Math.log1p(verdicts) / Math.log1p(400));
}

const ORIGIN = [0.1, 0.84];
const TARGET = [0.66, 0.46];

function geometry(width, height, verdicts) {
  const p = verdicts == null ? 0 : travelled(verdicts);
  const settled = verdicts == null ? 0.25 : settledness(verdicts);
  const ox = ORIGIN[0] * width;
  const oy = ORIGIN[1] * height;
  const base = Math.min(width, height) * 0.12;
  // Tightens as it settles: a model that has been told little draws a wide,
  // uncertain circle around where it thinks you are.
  const inner = base * (1.6 - settled);
  const outer = inner * 1.6;
  // Kept inside the stage, so early on the rings are not cut off by its edge.
  const yx = Math.max(outer + 8, ox + (TARGET[0] * width - ox) * p);
  const yy = Math.min(height - outer - 8, oy + (TARGET[1] * height - oy) * p);
  return { ox, oy, yx, yy, inner, outer };
}

/**
 * Where your taste sits among everything the catalogue knows.
 *
 * The points are laid out by a seeded generator rather than by a real
 * projection: the engine has no 2-D embedding yet, and inventing one that
 * looked meaningful would be a lie. What is real is bound to real numbers —
 * how far the path reaches and how tight the inner ring is both come from the
 * verdict count. The labels say which is which.
 */
/**
 * `pending` is for while the model is loading or cannot answer: only the
 * catalogue is drawn, since showing the cold start to someone with hundreds
 * of verdicts would tell them something false.
 */
export default function TasteField({ verdicts = null, pending = false, catalogue = null }) {
  const canvasRef = useRef(null);
  const mouse = useRef([-999, -999]);
  const [size, setSize] = useState([0, 0]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let frame;
    let width = 0;
    let height = 0;
    let far = [];
    let near = [];
    let t = 0;

    function resize() {
      const box = canvas.getBoundingClientRect();
      width = box.width;
      height = box.height;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      setSize([width, height]);

      // Seeded, so the scene is the same every visit.
      let seed = 7;
      const rand = () => {
        seed = (seed * 1664525 + 1013904223) % 4294967296;
        return seed / 4294967296;
      };
      far = Array.from({ length: Math.round((width * height) / 900) }, () => {
        const angle = rand() * Math.PI * 2;
        const radius = Math.sqrt(rand());
        return {
          x: width / 2 + Math.cos(angle) * radius * width * 0.49,
          y: height / 2 + Math.sin(angle) * radius * height * 0.47,
          r: 0.5 + rand() * 0.8,
          p: rand() * Math.PI * 2,
          s: 0.2 + rand() * 0.5,
          a: 1.5 + rand() * 3.5,
        };
      });
      near = Array.from({ length: 140 }, () => ({
        angle: rand() * Math.PI * 2,
        radius: Math.pow(rand(), 0.7) * 0.85,
        r: 0.9 + rand() * 0.9,
        p: rand() * Math.PI * 2,
        s: 0.3 + rand() * 0.6,
      }));
      if (still) draw();
    }

    function draw() {
      if (!still) t += 0.016;
      ctx.clearRect(0, 0, width, height);
      const g = geometry(width, height, verdicts);
      const [mx, my] = mouse.current;
      const breathe = 1 + 0.03 * Math.sin(t * 0.6);

      for (const star of far) {
        const x = star.x + Math.cos(t * star.s + star.p) * star.a;
        const y = star.y + Math.sin(t * star.s * 0.8 + star.p) * star.a;
        const twinkle = 0.5 + 0.5 * Math.sin(t * star.s * 2 + star.p);
        const hover = mx > -500 ? Math.max(0, 1 - Math.hypot(x - mx, y - my) / 110) : 0;
        ctx.beginPath();
        ctx.arc(x, y, star.r + hover * 1.2, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255,255,255,${Math.min(0.9, 0.1 + twinkle * 0.14 + hover * 0.55).toFixed(3)})`;
        ctx.fill();
      }

      if (pending) {
        if (!still) frame = requestAnimationFrame(draw);
        return;
      }

      ctx.beginPath();
      ctx.arc(g.ox, g.oy, 3, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(255,255,255,0.45)";
      ctx.fill();

      // Nothing learned yet: no path, rings or neighbourhood, because there
      // is no taste to be near. The start point pulses instead.
      if (verdicts == null) {
        const age = (t * 0.4) % 1;
        ctx.beginPath();
        ctx.arc(g.ox, g.oy, 4 + age * 26, 0, Math.PI * 2);
        ctx.lineWidth = 1;
        ctx.strokeStyle = `rgba(255,255,255,${(0.35 * (1 - age)).toFixed(3)})`;
        ctx.stroke();
        if (!still) frame = requestAnimationFrame(draw);
        return;
      }

      const halo = ctx.createRadialGradient(g.yx, g.yy, 0, g.yx, g.yy, g.outer * 1.4);
      halo.addColorStop(0, "rgba(255,255,255,0.07)");
      halo.addColorStop(1, "rgba(255,255,255,0)");
      ctx.fillStyle = halo;
      ctx.beginPath();
      ctx.arc(g.yx, g.yy, g.outer * 1.4, 0, Math.PI * 2);
      ctx.fill();

      // The path from knowing nothing to where it thinks you are.
      const edge = Math.max(0, Math.hypot(g.yx - g.ox, g.yy - g.oy) - g.inner * breathe);
      const len = Math.hypot(g.yx - g.ox, g.yy - g.oy) || 1;
      const ex = g.ox + ((g.yx - g.ox) / len) * edge;
      const ey = g.oy + ((g.yy - g.oy) / len) * edge;
      ctx.setLineDash([3, 5]);
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(255,255,255,0.28)";
      ctx.beginPath();
      ctx.moveTo(g.ox, g.oy);
      ctx.lineTo(ex, ey);
      ctx.stroke();
      ctx.setLineDash([]);
      if (edge > 0) {
        for (let k = 0; k < 3; k += 1) {
          const f = (t * 0.12 + k / 3) % 1;
          ctx.beginPath();
          ctx.arc(g.ox + (ex - g.ox) * f, g.oy + (ey - g.oy) * f, 1.6, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(255,255,255,${(0.7 * Math.sin(f * Math.PI)).toFixed(3)})`;
          ctx.fill();
        }
      }

      for (const [radius, alpha] of [[g.outer, 0.13], [g.inner, 0.34]]) {
        ctx.beginPath();
        ctx.arc(g.yx, g.yy, radius * breathe, 0, Math.PI * 2);
        ctx.lineWidth = 1;
        ctx.strokeStyle = `rgba(255,255,255,${alpha})`;
        ctx.stroke();
      }

      for (const dot of near) {
        const angle = dot.angle + Math.sin(t * dot.s * 0.3 + dot.p) * 0.12;
        const radius = dot.radius * g.inner * breathe;
        const x = g.yx + Math.cos(angle) * radius;
        const y = g.yy + Math.sin(angle) * radius;
        const twinkle = 0.6 + 0.4 * Math.sin(t * dot.s * 2 + dot.p);
        const hover = mx > -500 ? Math.max(0, 1 - Math.hypot(x - mx, y - my) / 110) : 0;
        ctx.beginPath();
        ctx.arc(x, y, dot.r + hover * 1.2, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255,255,255,${Math.min(1, 0.55 * twinkle + 0.2 + hover * 0.3).toFixed(3)})`;
        ctx.fill();
      }

      if (!still) frame = requestAnimationFrame(draw);
    }

    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    if (!still) frame = requestAnimationFrame(draw);

    const onMove = (event) => {
      const box = canvas.getBoundingClientRect();
      mouse.current = [event.clientX - box.left, event.clientY - box.top];
      if (still) draw();
    };
    const onLeave = () => {
      mouse.current = [-999, -999];
      if (still) draw();
    };
    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mouseleave", onLeave);

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      canvas.removeEventListener("mousemove", onMove);
      canvas.removeEventListener("mouseleave", onLeave);
    };
  }, [verdicts, pending]);

  const [width, height] = size;
  const g = width ? geometry(width, height, verdicts) : null;
  const told = verdicts ?? 0;

  return (
    <figure className="taste-field">
      <div className="taste-field-stage">
        <canvas
          ref={canvasRef}
          role="img"
          aria-label={
            pending
              ? "The catalogue as a field of points."
              : verdicts == null
              ? "The catalogue as a field of points, with you still at the starting point."
              : `The catalogue as a field of points. A path runs from where it started to a ring of titles closest to your taste, drawn from ${told} verdicts.`
          }
        />
        {g ? (
          <>
            <span className="taste-tag taste-tag-cloud" style={{ left: "4%", top: "5%" }}>
              everything in the catalogue
              {catalogue ? <b className="num"> · {catalogue.toLocaleString()} titles</b> : null}
            </span>
            {pending ? null : (
              <span className="taste-tag taste-tag-origin" style={{ left: g.ox, top: g.oy + 14 }}>
                where it started — knowing nothing about you
              </span>
            )}
            {pending ? null : verdicts == null ? (
              <span className="taste-tag taste-tag-start" style={{ left: g.ox + 36, top: g.oy }}>
                rate a few titles and it starts to move
              </span>
            ) : (
              <>
                <span
                  className="taste-tag taste-tag-path"
                  style={
                    // A short path puts its midpoint inside the rings, so the
                    // label moves above the start point instead.
                    Math.hypot(g.yx - g.ox, g.yy - g.oy) - g.outer < 120
                      ? { left: g.ox - 14, top: g.oy - 14, transform: "translateY(-100%)" }
                      : { left: (g.ox + g.yx) / 2, top: (g.oy + g.yy) / 2 }
                  }
                >
                  what <b className="num">{told.toLocaleString()}</b> verdicts taught it
                </span>
                <span className="taste-tag taste-tag-inner" style={{ left: g.yx, top: g.yy - g.inner - 10 }}>
                  titles closest to your taste
                </span>
                <span className="taste-tag taste-tag-outer" style={{ left: g.yx + g.outer + 10, top: g.yy }}>
                  near misses — worth a look
                </span>
              </>
            )}
          </>
        ) : null}
      </div>
      <figcaption>
        The distance travelled and how tight the inner ring is come from how many verdicts you
        have given. Where the points sit does not: this is a picture of scale, not a map. A real
        map of the catalogue is coming.
      </figcaption>
    </figure>
  );
}
