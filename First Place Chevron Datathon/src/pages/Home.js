// import React, { useState } from "react";

// const Home = () => {
//   return (
//     <div style={styles.container}>
//       {/* Mission Statement */}
//       <div style={styles.missionSection}>
//         <h1 style={styles.heading}>Our Mission</h1>
//         <p style={styles.missionText}>
//           At <strong>'Gas not Found'</strong>, we predict the future of cars by analyzing data from 2019 to 2024. 
//           Whether it's electric, hybrid, or gas vehicles, we forecast 2025 trends to help everyone prepare. 
//           Join us in exploring the future of driving, one data point at a time! 🚗✨
//         </p>
//       </div>

//       {/* User Stories Section */}
//       <div style={styles.userStoriesSection}>
//         <h2 style={styles.subHeading}>User Stories</h2>
//         <div style={styles.cardsContainer}>
//           <UserStoryCard 
//             image="/images/userstory1.jpg"
//             title="Emma Peterson"
//             text="The predictions from the Gas not Found project have been invaluable in understanding future transportation trends. It's empowering to see how data can guide us towards more sustainable vehicle choices."
//           />

//           <UserStoryCard 
//             image="/images/userstory2.jpg"
//             title="Frank Williams"
//             text="With the data from the Gas not Found project, I feel more confident in choosing safe and reliable vehicles for my family. It’s great to have a resource that considers our needs and budget."
//           />

//           <UserStoryCard 
//             image="/images/userstory3.jpg"
//             title="Isabella Cortez"
//             text="The Gas not Found project has made it easier for me to find cost-effective and fuel-efficient vehicles. It’s a relief to have access to reliable information that helps me manage my finances better."
//           />
//         </div>
//       </div>
//     </div>
//   );
// };

// /* ✅ Separate the UserStoryCard component to handle hover state */
// const UserStoryCard = ({ image, title, text }) => {
//   const [isHovered, setIsHovered] = useState(false);

//   return (
//     <div
//       style={{
//         ...styles.card,
//         transform: isHovered ? "scale(1.05)" : "scale(1)", // Zoom effect
//         boxShadow: isHovered ? "0px 6px 15px rgba(0, 0, 0, 0.3)" : styles.card.boxShadow,
//         background: isHovered ? "rgba(255, 255, 255, 0.5)" : styles.card.background, // Slightly lighter on hover
//       }}
//       onMouseEnter={() => setIsHovered(true)}
//       onMouseLeave={() => setIsHovered(false)}
//     >
//       <img 
//         src={image} 
//         alt={title} 
//         style={{
//           ...styles.cardImage,
//           transform: isHovered ? "scale(1.1)" : "scale(1)", // Enlarge image slightly
//         }} 
//       />
//       <h3 style={styles.cardTitle}>{title}</h3>
//       <p style={styles.cardText}>{text}</p>
//     </div>
//   );
// };

// /* ✅ Styles */
// const styles = {
//   container: {
//     display: "flex",
//     flexDirection: "column",
//     alignItems: "center",
//     justifyContent: "center",
//     padding: "40px 20px",
//     background: "linear-gradient(to bottom, #0057A0, #E0E0E0)", // Chevron Blue to Light Gray Gradient
//     minHeight: "100vh",
//   },
//   missionSection: {
//     textAlign: "center",
//     maxWidth: "700px",
//     padding: "30px",
//     marginTop: "100px",
//     background: "rgba(255, 255, 255, 0.15)", // Glassmorphism effect
//     backdropFilter: "blur(10px)",
//     boxShadow: "0px 4px 12px rgba(0, 0, 0, 0.2)",
//     borderRadius: "12px",
//     color: "#fff",
//   },
//   heading: {
//     fontSize: "36px",
//     fontWeight: "bold",
//     color: "#FFFFFF",
//     textShadow: "1px 1px 5px rgba(0, 0, 0, 0.2)",
//     marginBottom: "10px",
//   },
//   missionText: {
//     fontSize: "18px",
//     lineHeight: "1.6",
//   },
//   userStoriesSection: {
//     width: "100%",
//     maxWidth: "1200px",
//     textAlign: "center",
//     marginTop: "50px",
//   },
//   subHeading: {
//     fontSize: "28px",
//     fontWeight: "bold",
//     color: "#FFFFFF",
//     textShadow: "1px 1px 5px rgba(0, 0, 0, 0.2)",
//     marginBottom: "20px",
//   },
//   cardsContainer: {
//     display: "flex",
//     justifyContent: "center",
//     gap: "25px",
//     flexWrap: "wrap",
//   },
//   card: {
//     width: "320px",
//     background: "rgba(255, 255, 255, 0.2)", // Glassmorphism effect
//     backdropFilter: "blur(12px)",
//     padding: "25px",
//     borderRadius: "14px",
//     boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.1)",
//     textAlign: "center",
//     transition: "transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out, background 0.3s ease-in-out",
//     color: "rgba(14, 3, 3, 0.86)", // Ensuring white text for better visibility
//   },
//   cardImage: {
//     width: "140px", // Increased size from 100px
//     height: "140px",
//     borderRadius: "50%",
//     marginBottom: "15px",
//     transition: "transform 0.3s ease-in-out",
//   },
//   cardTitle: {
//     fontSize: "20px",
//     fontWeight: "bold",
//     color: "#E21737", // Chevron Red
//     marginBottom: "10px",
//   },
//   cardText: {
//     fontSize: "16px",
//     lineHeight: "1.5",
//   },
// };

// export default Home;
// TRIAL CODE

// import React from "react";
// import Particles from "react-tsparticles";
// import { loadFull } from "tsparticles";

// const Home = () => {
//   const particlesInit = async (main) => {
//     await loadFull(main);
//   };

//   return (
//     <div style={styles.container}>
//       {/* Particles Background */}
//       <Particles 
//         init={particlesInit}
//         options={{
//           background: { color: "#0057A0" }, // Chevron Blue
//           particles: {
//             number: { value: 100 },
//             size: { value: 3 },
//             move: { speed: 2 },
//             links: { enable: true, color: "#ffffff" },
//           },
//         }}
//         style={styles.particles}
//       />

//       {/* Mission Section */}
//       <div style={styles.missionSection}>
//         <h1 style={styles.heading}>Our Mission</h1>
//         <p style={styles.missionText}>
//           At <strong>'Gas not Found'</strong>, we predict the future of cars by analyzing data from 2019 to 2024. 
//           Whether it's electric, hybrid, or gas vehicles, we forecast 2025 trends to help everyone prepare. 
//           Join us in exploring the future of driving, one data point at a time! 🚗✨
//         </p>
//       </div>
//     </div>
//   );
// };

// /* ✅ Styles */
// const styles = {
//   container: {
//     position: "relative",
//     width: "100%",
//     minHeight: "100vh",
//     display: "flex",
//     flexDirection: "column",
//     alignItems: "center",
//     justifyContent: "center",
//     overflow: "hidden",
//   },
//   particles: {
//     position: "absolute",
//     top: 0,
//     left: 0,
//     width: "100%",
//     height: "100%",
//     zIndex: -1,
//   },
//   missionSection: {
//     textAlign: "center",
//     maxWidth: "700px",
//     padding: "30px",
//     background: "rgba(0, 0, 0, 0.6)", // Dark background for readability
//     borderRadius: "12px",
//     color: "#fff",
//     zIndex: 1,
//   },
//   heading: {
//     fontSize: "36px",
//     fontWeight: "bold",
//   },
//   missionText: {
//     fontSize: "18px",
//     lineHeight: "1.6",
//   },

  
// };

// export default Home;

import React, { useState, useCallback } from "react";
import Particles from "react-tsparticles";
import { loadSlim } from "tsparticles-slim";

const Home = () => {
  const particlesInit = useCallback(async (engine) => {
    await loadSlim(engine);
  }, []);

  return (
    <div style={styles.container}>
      {/* Particles Background */}
      <Particles 
        init={particlesInit}
        options={{
          background: { color: "#0057A0" },
          particles: {
            number: { value: 80 },
            size: { value: 3 },
            move: { speed: 2 },
            links: { enable: true, color: "#ffffff" },
          },
        }}
        style={styles.particles}
      />

      {/* Mission Statement */}
      <div style={styles.missionSection}>
        <h1 style={styles.heading}>Our Mission</h1>
        <p style={styles.missionText}>
          At <strong>'Gas not Found'</strong>, we predict the future of cars by analyzing data from 2019 to 2023. 
          Whether it's electric, hybrid, or gas vehicles, we forecast 2024 trends to help everyone prepare. 
          Join us in exploring the future of driving, one data point at a time! 🚗✨
        </p>
      </div>

      {/* User Stories Section */}
      <div style={styles.userStoriesSection}>
        <h2 style={styles.subHeading}>User Stories</h2>
        <div style={styles.cardsContainer}>
          <UserStoryCard 
            image="/images/userstory1.jpg"
            title="Emma Peterson"
            text={<em>"The predictions from the Gas not Found project have been invaluable in understanding future transportation trends."</em>}
            />

          <UserStoryCard 
            image="/images/userstory2.jpg"
            title="Frank Williams"
            text={<em>"With the data from the Gas not Found project, I feel more confident in choosing safe and reliable vehicles for my family."</em>}
          />

          <UserStoryCard 
            image="/images/userstory3.jpg"
            title="Isabella Cortez"
            text={<em>"The Gas not Found project has made it easier for me to find cost-effective and fuel-efficient vehicles."</em>}
          />
        </div>
      </div>
    </div>
  );
};

/* ✅ User Story Card Component */
const UserStoryCard = ({ image, title, text }) => {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div
      style={{
        ...styles.card,
        transform: isHovered ? "scale(1.05)" : "scale(1)",
        boxShadow: isHovered ? "0px 6px 15px rgba(0, 0, 0, 0.3)" : styles.card.boxShadow,
        background: isHovered ? "rgba(255, 255, 255, 0.5)" : styles.card.background,
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <img 
        src={image} 
        alt={title} 
        style={{
          ...styles.cardImage,
          transform: isHovered ? "scale(1.1)" : "scale(1)",
        }} 
      />
      <h3 style={styles.cardTitle}>{title}</h3>
      <p style={styles.cardText}>{text}</p>
    </div>
  );
};

/* ✅ Styles */
const styles = {
  container: {
    position: "relative",
    width: "100%",
    minHeight: "100vh",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    overflow: "hidden",
  },
  particles: {
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
    zIndex: -1,
  },
  missionSection: {
    textAlign: "center",
    maxWidth: "700px",
    padding: "30px",
    marginTop: "120px",  // ⬅️ Pushes the mission section down to avoid overlapping
    background: "rgba(0, 0, 0, 0.6)", // Dark overlay for readability
    borderRadius: "12px",
    color: "#fff",
    zIndex: 1,
  },
  
  heading: {
    fontSize: "36px",
    fontWeight: "bold",
  },
  missionText: {
    fontSize: "18px",
    lineHeight: "1.6",
  },
  userStoriesSection: {
    width: "100%",
    maxWidth: "1200px",
    textAlign: "center",
    marginTop: "50px",
    paddingBottom: "50px",
  },
  subHeading: {
    fontSize: "28px",
    fontWeight: "bold",
    color: "#FFFFFF",
    textShadow: "1px 1px 5px rgba(0, 0, 0, 0.5)",
    marginBottom: "20px",
  },
  cardsContainer: {
    display: "flex",
    justifyContent: "center",
    gap: "25px",
    flexWrap: "wrap",
  },
  card: {
    width: "320px",
    background: "rgba(255, 255, 255, 0.2)", // Glassmorphism effect
    backdropFilter: "blur(12px)",
    padding: "25px",
    borderRadius: "14px",
    boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.1)",
    textAlign: "center",
    transition: "transform 0.3s ease-in-out, box-shadow 0.3s ease-in-out, background 0.3s ease-in-out",
    color: "#fff",
  },
  cardImage: {
    width: "140px",
    height: "140px",
    borderRadius: "50%",
    marginBottom: "15px",
    transition: "transform 0.3s ease-in-out",
    objectFit: "cover",
  },
  cardTitle: {
    fontSize: "20px",
    fontWeight: "bold",
    color: "#FFD700", // Gold for contrast
    marginBottom: "10px",
  },
  cardText: {
    fontSize: "16px",
    lineHeight: "1.5",
  },
};

export default Home;
