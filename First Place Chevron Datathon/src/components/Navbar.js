import { Link } from "react-router-dom";

const Navbar = () => {
  return (
    <nav style={styles.navbar}>
      {/* Logo + Project Name */}
      <div style={styles.logoContainer}>
        <img src="C:\Users\rajai\gasnotfound-1\public\images\userstory1.jpg" alt="Gas Not Found Logo" style={styles.logoImage} />
        <div style={styles.logoText}>404. Gas Not Found</div>
      </div>

      {/* Navigation Links */}
      <ul style={styles.navLinks}>
        <li><Link to="/" style={styles.navLink}>Home</Link></li>
        <li><Link to="/prediction" style={styles.navLink}>Prediction</Link></li>
        <li><Link to="/about" style={styles.navLink}>About</Link></li>
      </ul>
    </nav>
  );
};

const styles = {
  navbar: {
    display: "flex",
    justifyContent: "space-between", 
    alignItems: "center",
    padding: "12px 30px",
    backgroundColor: "rgba(255, 255, 255, 0.9)", // Light & Transparent
    position: "fixed",
    top: 0,
    width: "100%",
    zIndex: 1000,
    boxShadow: "0px 2px 8px rgba(0, 0, 0, 0.1)",
  },
  logoContainer: {
    display: "flex",
    alignItems: "center",
    gap: "10px", // Adds spacing between logo and text
  },
  logoImage: {
    width: "50px",  // Slightly larger logo
    height: "50px",
    borderRadius: "50%", // Keeps it circular
  },
  logoText: {
    fontSize: "22px",  // Slightly bigger for visibility
    fontWeight: "bold",
    color: "#0057A0",
    fontFamily: "'Orbitron', sans-serif", // ✅ Google Font applied
    letterSpacing: "1px", // Adds a futuristic style
    textTransform: "uppercase", // Makes it look more like a logo
  },
  navLinks: {
    listStyle: "none",
    display: "flex",
    gap: "20px",
    marginRight: "90px", // Adjusts position (not too far right)
  },
  navLink: {
    textDecoration: "none",
    color: "#333",
    fontSize: "18px",
    fontWeight: "500",
    transition: "color 0.3s ease-in-out",
  },
};

export default Navbar;
