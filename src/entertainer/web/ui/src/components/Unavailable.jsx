/**
 * One way of saying "not yet", wherever a page cannot render.
 *
 * A half-built machine is a normal state for hours. The server already
 * phrases each of those states for a reader rather than for a terminal — see
 * web/failures.py — so this shows that sentence and adds nothing to it. Two
 * paragraphs saying the same thing read as an error stack.
 */
export default function Unavailable({ error, children }) {
  return (
    <div className="empty">
      <p>{error?.message || "This did not load."}</p>
      {children}
    </div>
  );
}
