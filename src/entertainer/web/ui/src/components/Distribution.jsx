/**
 * A prediction drawn as the distribution it actually is.
 *
 * The engine computes a posterior mean and a predictive standard deviation
 * for every title and then throws the second away into "± 2.2". Drawing the
 * curve costs nothing extra and is the honest rendering: a model that barely
 * knows you produces a wide low hill, one that is sure produces a spike.
 */
export default function Distribution({ score, std, height = 44 }) {
  if (score == null || Number.isNaN(score)) return null;

  const width = 440;
  const spread = Math.max(0.35, Math.min(std ?? 2, 4));
  const centre = (Math.max(0, Math.min(10, score)) / 10) * width;
  const sigma = (spread / 10) * width;

  // Sampled rather than a bezier guess, so the drawn shape is the Gaussian
  // the numbers describe.
  const points = [];
  for (let x = 0; x <= width; x += 4) {
    const z = (x - centre) / sigma;
    points.push([x, Math.exp(-0.5 * z * z)]);
  }
  const path = points
    .map(([x, y], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)},${(height - y * (height - 3)).toFixed(1)}`)
    .join(" ");

  return (
    <div className="dist">
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img"
           aria-label={`predicted ${score.toFixed(1)} out of 10, give or take ${(std ?? 0).toFixed(1)}`}>
        <path d={`${path} L${width},${height} L0,${height} Z`} className="dist-fill" />
        <path d={path} className="dist-line" />
        <line x1={centre} y1="0" x2={centre} y2={height} className="dist-mark" />
      </svg>
      <div className="dist-scale label"><span>0</span><span>5</span><span>10</span></div>
    </div>
  );
}
