import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Impact from "./pages/Home";
import Prediction from "./pages/Prediction";
import About from "./pages/About";

const App = () => {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<Impact />} />
        <Route path="/prediction" element={<Prediction />} />
        <Route path="/about" element={<About />} />
      </Routes>
    </Router>
  );
};

export default App;
