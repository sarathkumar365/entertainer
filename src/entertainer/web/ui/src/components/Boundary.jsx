import { Component } from "react";

/**
 * Keep one broken page from blanking the whole interface.
 *
 * Without this, a render error unmounts the root: the tab goes white and
 * even the nav is gone, so there is no way back to a tab that still works.
 * React offers no hook equivalent, hence the class.
 */
export default class Boundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error) {
    // The console is where a developer looks; the page says the short version.
    console.error(error);
  }

  componentDidUpdate(prev) {
    // A new route is a new attempt: clearing here is what makes the other
    // tabs reachable again after one of them throws.
    if (prev.resetKey !== this.props.resetKey && this.state.error) {
      this.setState({ error: null });
    }
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="page">
        <div className="empty">
          <p>This page could not be drawn.</p>
          <p>The other tabs still work. Reload to try this one again.</p>
        </div>
      </div>
    );
  }
}
