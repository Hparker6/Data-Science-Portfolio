import React, { useCallback } from "react";
import Particles from "react-tsparticles";
import { loadSlim } from "tsparticles-slim";

const teamMembers = [
  {
    name: "Gaurav Govindwani",
    university: "University of North Texas",
    linkedin: "https://www.linkedin.com/in/gauravgovindwani",
    image: "/images/gaurav_logo.jpg",
  },
  {
    name: "Maya Espinoza",
    university: "Georgia Tech University",
    linkedin: "https://www.linkedin.com/in/mayae-spinosa/",
    image: "/images/maya2_logo.jpg",
  },
  {
    name: "Houston Parker",
    university: "University of North Texas",
    linkedin: "http://www.linkedin.com/in/houston-parkerr",
    image: "/images/houston_logo.jpg",
  },
  {
    name: "Insiya Raja",
    university: "Texas State University",
    linkedin: "https://www.linkedin.com/in/insiya-raja24/",
    image: "/images/insiya_logo.jpg",
  },
];

const About = () => {
  const particlesInit = useCallback(async (engine) => {
    await loadSlim(engine);
  }, []);

  return (
    <div style={styles.container}>
      {/* Particles Background */}
      <Particles 
        init={particlesInit}
        options={{
          background: { color: "#1E1E1E" }, // Darker background for contrast
          particles: {
            number: { value: 100 },
            size: { value: 3 },
            move: { speed: 1.5 },
            links: { enable: true, color: "#FFD700" }, // Gold particle lines
          },
        }}
        style={styles.particles}
      />

      {/* Funky Intro */}
      <div style={styles.header}>
        <h1 style={styles.title}>🚀 Who Are We? 🚀</h1>
        <p style={styles.description}>
          We are **Computer Science & Data Science students** at this **Hackathon**,  
          pushing the limits of **data analytics** to predict **vehicle trends for 2025**!  
        </p>
      </div>

      {/* Team Section */}
      <div style={styles.teamSection}>
        {teamMembers.map((member, index) => (
          <div key={index} style={styles.card}>
            <a href={member.linkedin} target="_blank" rel="noopener noreferrer">
              <img src={member.image} alt={member.name} style={styles.profileImage} />
            </a>
            <h3 style={styles.name}>{member.name}</h3>
            <p style={styles.university}>{member.university}</p>
            <a href={member.linkedin} target="_blank" rel="noopener noreferrer" style={styles.linkedinLink}>
              Connect on LinkedIn
            </a>
          </div>
        ))}
      </div>
    </div>
  );
};

/* 🔹 Styling */
const styles = {
  container: {
    position: "relative",
    textAlign: "center",
    padding: "10px 10px",
    background: "#1E1E1E", // Dark background for contrast
    color: "#fff",
    minHeight: "100vh",
  },
  particles: {
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    zIndex: -1, // Ensures particles stay in the background
  },
  header: {
    maxWidth: "800px",
    // margin: "auto 5px",  // ⬅️ Reduced margin from 100px to 70px
    padding: "20px",
    borderRadius: "10px",
    background: "rgba(0, 0, 0, 0.5)", // Overlay for readability
  },
  
  title: {
    fontSize: "36px",
    fontWeight: "bold",
    textShadow: "1px 1px 5px rgba(0, 0, 0, 0.3)",
  },
  description: {
    fontSize: "18px",
    lineHeight: "1.6",
    marginBottom: "40px",
  },
  teamSection: {
    display: "flex",
    justifyContent: "center",
    gap: "20px",
    flexWrap: "wrap",
    marginTop: "-40px",  // ⬅️ Reduced from default spacing
  },
  
  card: {
    width: "220px",
    background: "rgba(255, 255, 255, 0.2)", // Glassmorphism effect
    backdropFilter: "blur(10px)",
    padding: "20px",
    borderRadius: "12px",
    boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.2)",
    textAlign: "center",
    transition: "transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out",
  },
  profileImage: {
    width: "140px",
    height: "140px",
    borderRadius: "50%",
    marginBottom: "15px",
    border: "3px solid #fff",
    objectFit: "cover",
    objectPosition: "center",
    transition: "transform 0.3s ease-in-out",
  },
  name: {
    fontSize: "20px",
    fontWeight: "bold",
  },
  university: {
    fontSize: "16px",
    fontStyle: "italic",
    marginBottom: "10px",
  },
  linkedinLink: {
    fontSize: "14px",
    color: "#FFD700", // Gold color for LinkedIn link
    textDecoration: "none",
    fontWeight: "bold",
    transition: "color 0.3s ease-in-out",
  },
};

// **✅ Hover Effects**
styles.card[":hover"] = {
  transform: "scale(1.05)",
  boxShadow: "0px 6px 12px rgba(0, 0, 0, 0.3)",
};
styles.profileImage[":hover"] = {
  transform: "scale(1.1)", // Slightly enlarges on hover
};

export default About;
