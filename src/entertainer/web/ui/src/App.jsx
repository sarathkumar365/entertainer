import { NavLink, Route, Routes } from "react-router-dom";

import Evidence from "./pages/Evidence";
import Home from "./pages/Home";
import Rate from "./pages/Rate";
import Recommendations from "./pages/Recommendations";
import Taste from "./pages/Taste";

const TABS = [
  ["/recs", "Recommend"],
  ["/rate", "Rate"],
  ["/taste", "Taste"],
  ["/evidence", "Evidence"],
];

export default function App() {
  return (
    <div className="app">
      <nav className="nav">
        <NavLink to="/" className="brand">entertainer</NavLink>
        {TABS.map(([to, label]) => (
          <NavLink key={to} to={to} className="tab">{label}</NavLink>
        ))}
      </nav>
      <main>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/recs" element={<Recommendations />} />
          <Route path="/rate" element={<Rate />} />
          <Route path="/taste" element={<Taste />} />
          <Route path="/evidence" element={<Evidence />} />
          <Route path="*" element={<Home />} />
        </Routes>
      </main>
    </div>
  );
}
