import { useEffect, useRef } from "react";

/**
 * The taste model, drawn as a living thing.
 *
 * Every property is bound to a real quantity rather than chosen for effect:
 *
 *   radius     <- verdicts absorbed
 *   wobble     <- 1 - confidence; a model that is guessing visibly wavers
 *   ripple     <- a verdict landing
 *   reaching   <- how many titles are in play; `rated` of them answered
 *
 * Two things are ambient rather than bound, and should not be read as data:
 * the scattered field, which is texture at catalogue scale, and which
 * internal nodes light up, which is a fixed random weighting rather than the
 * actual heaviest axes. Wiring those would mean shipping the item space and
 * the weight vector to the browser for a background animation.
 *
 * Canvas rather than SVG because the field is a few hundred points redrawn
 * every frame, and because the membrane is a path recomputed each tick from
 * a sum of sines — neither is something the DOM should be asked to do.
 */
export default function Organism({
  verdicts = 0,
  confidence = 0.5,
  reaching = 0,
  rated = 0,
  working = false,
  pulseKey = 0,
  compact = false,
}) {
  const canvasRef = useRef(null);
  const state = useRef({ pulses: [], t: 0, mouse: [-999, -999] });

  // A pulse per verdict recorded, driven by a key the caller bumps.
  useEffect(() => {
    if (pulseKey > 0) state.current.pulses.push({ t0: state.current.t });
  }, [pulseKey]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let frame;
    let stars = [];
    let width = 0;
    let height = 0;

    const nodeCount = compact ? 10 : 20;
    const nodes = Array.from({ length: nodeCount }, (_, i) => ({
      a: (i / nodeCount) * Math.PI * 2 + Math.random() * 0.3,
      r: 0.22 + Math.random() * 0.64,
      w: 0.3 + Math.random() * 0.9,
      p: Math.random() * Math.PI * 2,
    }));

    const reachCount = Math.max(0, reaching);
    const reach = Array.from({ length: reachCount }, (_, i) => ({
      a: (i / Math.max(reachCount, 1)) * Math.PI * 2 + 0.45,
      d: 1.5 + Math.random() * 0.75,
      p: Math.random() * Math.PI * 2,
      rated: i < rated,
    }));

    function resize() {
      const box = canvas.getBoundingClientRect();
      width = box.width;
      height = box.height;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const density = compact ? 9000 : 2600;
      stars = Array.from({ length: Math.round((width * height) / density) }, () => ({
        x: Math.random() * width,
        y: Math.random() * height,
        r: 0.4 + Math.random() * 1.0,
        p: Math.random() * Math.PI * 2,
        s: 0.25 + Math.random() * 0.6,
      }));
    }
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);

    function radiusAt(angle, base, tight, time) {
      const wobble = 1 - tight;
      let r =
        base *
        (1 +
          0.085 * wobble * Math.sin(3 * angle + time * 0.55) +
          0.06 * wobble * Math.sin(5 * angle - time * 0.38) +
          0.038 * wobble * Math.sin(7 * angle + time * 0.72) +
          0.02 * Math.sin(11 * angle - time * 0.9));
      r *= 1 + 0.022 * Math.sin(time * 1.1);
      // While the engine is thinking, the membrane churns rather than a
      // spinner appearing somewhere else on the page.
      if (working) r *= 1 + 0.05 * Math.sin(time * 4.2 + angle * 2);
      for (const pulse of state.current.pulses) {
        const age = time - pulse.t0;
        const d = Math.abs(1 - age * 2.4);
        if (d < 0.5) r += base * 0.1 * Math.cos(d * Math.PI) * Math.max(0, 1 - age / 1.9);
      }
      return r;
    }

    function draw() {
      const s = state.current;
      s.t += 0.016;
      const t = s.t;
      s.pulses = s.pulses.filter((p) => t - p.t0 < 2);

      ctx.clearRect(0, 0, width, height);
      const cx = width / 2;
      const cy = height * (compact ? 0.5 : 0.42);
      const grow = Math.min(verdicts, 800) / 800;
      const base = Math.min(width, height) * ((compact ? 0.26 : 0.155) + grow * 0.075);
      const [mx, my] = s.mouse;

      for (const star of stars) {
        const dx = star.x - cx;
        const dy = star.y - cy;
        const near = Math.max(0, 1 - Math.hypot(dx, dy) / (base * 4.2));
        const twinkle = 0.5 + 0.5 * Math.sin(t * star.s + star.p);
        let hover = 0;
        if (mx > -500) hover = Math.max(0, 1 - Math.hypot(star.x - mx, star.y - my) / 90);
        ctx.beginPath();
        ctx.arc(star.x, star.y, star.r + hover * 1.1, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255,255,255,${Math.min(
          0.95,
          (0.05 + near * 0.32) * (0.55 + twinkle * 0.45) + hover * 0.5,
        ).toFixed(3)})`;
        ctx.fill();
      }

      const halo = ctx.createRadialGradient(cx, cy, base * 0.3, cx, cy, base * 2.6);
      halo.addColorStop(0, "rgba(255,255,255,0.055)");
      halo.addColorStop(1, "rgba(255,255,255,0)");
      ctx.fillStyle = halo;
      ctx.beginPath();
      ctx.arc(cx, cy, base * 2.6, 0, Math.PI * 2);
      ctx.fill();

      for (const line of reach) {
        const angle = line.a + Math.sin(t * 0.1 + line.p) * 0.05;
        const far = base * (line.d + Math.sin(t * 0.5 + line.p) * 0.05);
        const ex = cx + Math.cos(angle) * far;
        const ey = cy + Math.sin(angle) * far;
        const sr = radiusAt(angle, base, confidence, t);
        const sx = cx + Math.cos(angle) * sr;
        const sy = cy + Math.sin(angle) * sr;
        const flow = (Math.sin(t * 0.8 + line.p) + 1) / 2;
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(ex, ey);
        ctx.lineWidth = 0.8;
        ctx.strokeStyle = `rgba(255,255,255,${(0.06 + flow * 0.1).toFixed(3)})`;
        ctx.stroke();
        const travel = (t * 0.35 + line.p) % 1;
        ctx.beginPath();
        ctx.arc(sx + (ex - sx) * travel, sy + (ey - sy) * travel, 1.1, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255,255,255,${(0.5 * (1 - Math.abs(travel - 0.5) * 2) + 0.12).toFixed(3)})`;
        ctx.fill();
        ctx.beginPath();
        ctx.arc(ex, ey, line.rated ? 2.6 : 2, 0, Math.PI * 2);
        ctx.fillStyle = line.rated
          ? "rgba(214,232,150,0.85)"
          : `rgba(255,255,255,${(0.35 + flow * 0.35).toFixed(3)})`;
        ctx.fill();
      }

      const count = 120;
      const points = [];
      for (let i = 0; i < count; i += 1) {
        const angle = (i / count) * Math.PI * 2;
        const r = radiusAt(angle, base, confidence, t);
        points.push([cx + Math.cos(angle) * r, cy + Math.sin(angle) * r]);
      }
      const trace = () => {
        ctx.beginPath();
        ctx.moveTo(
          (points[0][0] + points[count - 1][0]) / 2,
          (points[0][1] + points[count - 1][1]) / 2,
        );
        for (let i = 0; i < count; i += 1) {
          const c = points[i];
          const n = points[(i + 1) % count];
          ctx.quadraticCurveTo(c[0], c[1], (c[0] + n[0]) / 2, (c[1] + n[1]) / 2);
        }
        ctx.closePath();
      };

      trace();
      const body = ctx.createRadialGradient(
        cx - base * 0.3, cy - base * 0.35, base * 0.05, cx, cy, base * 1.1,
      );
      body.addColorStop(0, "rgba(255,255,255,0.10)");
      body.addColorStop(1, "rgba(255,255,255,0.015)");
      ctx.fillStyle = body;
      ctx.fill();

      const placed = nodes.map((node) => {
        const angle = node.a + Math.sin(t * 0.18 * node.w + node.p) * 0.16;
        const r = radiusAt(angle, base, confidence, t) * node.r;
        return [cx + Math.cos(angle) * r, cy + Math.sin(angle) * r];
      });
      ctx.lineWidth = 0.6;
      for (let a = 0; a < placed.length; a += 1) {
        for (let b = a + 1; b < placed.length; b += 1) {
          const d = Math.hypot(placed[a][0] - placed[b][0], placed[a][1] - placed[b][1]);
          if (d < base * 0.6) {
            ctx.strokeStyle = `rgba(255,255,255,${((1 - d / (base * 0.6)) * 0.22).toFixed(3)})`;
            ctx.beginPath();
            ctx.moveTo(placed[a][0], placed[a][1]);
            ctx.lineTo(placed[b][0], placed[b][1]);
            ctx.stroke();
          }
        }
      }
      placed.forEach((point, i) => {
        const flicker = 0.4 + 0.6 * Math.abs(Math.sin(t * nodes[i].w + nodes[i].p));
        const hot = nodes[i].w > 0.95;
        ctx.beginPath();
        ctx.arc(point[0], point[1], hot ? 2 : 1.3, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255,255,255,${(hot ? flicker : flicker * 0.45).toFixed(3)})`;
        ctx.fill();
      });

      trace();
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = `rgba(255,255,255,${(0.22 + confidence * 0.5).toFixed(3)})`;
      ctx.stroke();

      for (const pulse of s.pulses) {
        const age = t - pulse.t0;
        const fade = Math.max(0, 1 - age / 2);
        ctx.beginPath();
        ctx.arc(cx, cy, base * (0.5 + age * 1.7), 0, Math.PI * 2);
        ctx.lineWidth = 1;
        ctx.strokeStyle = `rgba(255,255,255,${(fade * 0.18).toFixed(3)})`;
        ctx.stroke();
      }

      frame = requestAnimationFrame(draw);
    }
    frame = requestAnimationFrame(draw);

    const onMove = (event) => {
      const box = canvas.getBoundingClientRect();
      state.current.mouse = [event.clientX - box.left, event.clientY - box.top];
    };
    const onLeave = () => {
      state.current.mouse = [-999, -999];
    };
    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mouseleave", onLeave);

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      canvas.removeEventListener("mousemove", onMove);
      canvas.removeEventListener("mouseleave", onLeave);
    };
  }, [verdicts, confidence, reaching, rated, working, compact]);

  return <canvas ref={canvasRef} className="organism" aria-hidden="true" />;
}
