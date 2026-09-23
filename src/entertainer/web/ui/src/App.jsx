import { NavLink, Route, Routes, useLocation } from "react-router-dom";

import Boundary from "./components/Boundary";
import Evidence from "./pages/Evidence";
import Home from "./pages/Home";
import Library from "./pages/Library";
import Rate from "./pages/Rate";
import Recommendations from "./pages/Recommendations";
import Taste from "./pages/Taste";

const TABS = [
  ["/recs", "Recommend"],
  ["/rate", "Rate"],
  ["/library", "Library"],
  ["/taste", "Taste"],
  ["/evidence", "Evidence"],
];

export default function App() {
  const location = useLocation();
  return (
    <div className="app">
      <nav className="nav">
        <NavLink to="/" className="brand">entertainer</NavLink>
        {TABS.map(([to, label]) => (
          <NavLink key={to} to={to} className="tab">{label}</NavLink>
        ))}
      </nav>
      <main>
        {/* Outside the nav on purpose: a page that throws must leave the
            tabs standing, or there is no way to reach one that works. */}
        <Boundary resetKey={location.pathname}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/recs" element={<Recommendations />} />
            <Route path="/rate" element={<Rate />} />
            <Route path="/library" element={<Library />} />
            <Route path="/taste" element={<Taste />} />
            <Route path="/evidence" element={<Evidence />} />
            <Route path="*" element={<Home />} />
          </Routes>
        </Boundary>
      </main>
    </div>
  );
}
