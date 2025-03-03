import React, { useCallback } from "react";
import Particles from "react-tsparticles";
import { loadSlim } from "tsparticles-slim";

const Prediction = () => {
  const particlesInit = useCallback(async (engine) => {
    await loadSlim(engine);
  }, []);

  // Graphs with Titles
  const graphs = [
    { src: "graph1.jpg", title: "📊 EV Adoption Trends" },
    { src: "graph2.jpg", title: "⛽ Fuel-Based Vehicle Decline" },
    { src: "graph3.jpg", title: "🔋 Hybrid Vehicles Growth" },
    { src: "graph4.jpg", title: "🌍 Regional Market Distribution" },
    { src: "graph5.jpg", title: "🌱 Emission Reductions by 2025" },
    { src: "graph6.jpg", title: "🚗 Consumer Preference Shift" },
  ];

  return (
    <div style={styles.container}>
      {/* Particles Background */}
      <Particles 
        init={particlesInit}
        options={{
          background: { color: "#0F2027" },
          particles: {
            number: { value: 100 },
            size: { value: 3 },
            move: { speed: 1.5 },
            color: { value: ["#00FFFF", "#FFD700", "#00BFFF"] },
            links: { enable: true, color: "#FFD700" },
          },
        }}
        style={styles.particles}
      />

      {/* Prediction Content */}
      <div style={styles.predictionSection}>
        <h1 style={styles.heading}> Vehicle Prediction</h1>
        <p style={styles.description}>
          Using advanced data analytics & machine learning, we predict the number of vehicles by 2025.  
          Discover insights into the future of EVs, hybrids, and fuel-based vehicles.
        </p>

        {/* Graphs Section */}
        <div style={styles.graphSection}>
          <h2 style={styles.graphHeading}>📊 Data Exploration & Analysis</h2>
          <div style={styles.graphGrid}>
            {graphs.map((graph, index) => (
              <div key={index} style={styles.graphCard}>
                <img 
                  src={`/images/${graph.src}`} 
                  alt={graph.title} 
                  style={styles.graphImage}
                />
                <p style={styles.graphCaption}>{graph.title}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

/* 🔹 Styles */
const styles = {
  container: {
    position: "relative",
    textAlign: "center",
    padding: "50px 20px",
    background: "#0F2027",
    color: "#fff",
    minHeight: "100vh",
  },
  particles: {
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    zIndex: -1,
  },
  predictionSection: {
    maxWidth: "1000px", // Increased width for better alignment
    margin: "80px auto 0",
    padding: "40px",
    borderRadius: "12px",
    background: "rgba(0, 0, 0, 0.5)",
    backdropFilter: "blur(10px)",
  },
  heading: {
    fontSize: "38px",
    fontWeight: "bold",
    color: "#FFD700",
    textShadow: "1px 1px 5px rgba(255, 215, 0, 0.5)",
  },
  description: {
    fontSize: "20px",
    lineHeight: "1.6",
    marginBottom: "30px",
  },
  graphSection: {
    marginTop: "30px",
  },
  graphHeading: {
    fontSize: "26px",
    fontWeight: "bold",
    color: "#00FFFF", // Light Blue accent
    marginBottom: "20px",
  },
  graphGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(2, 1fr)", // ⬅️ Displays 2 graphs per row
    gap: "25px",
    justifyContent: "center",
    marginTop: "20px",
  },
  graphCard: {
    background: "rgba(255, 255, 255, 0.1)",
    padding: "15px",
    borderRadius: "12px",
    boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.3)",
    textAlign: "center",
  },
  graphImage: {
    width: "100%",  // ⬅️ Makes the image fit perfectly
    height: "auto",
    borderRadius: "8px",
  },
  graphCaption: {
    fontSize: "18px",
    marginTop: "10px",
    fontWeight: "bold",
    color: "#FFD700", // Gold text
  },
};

export default Prediction;
