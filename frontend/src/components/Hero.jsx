import React, { useEffect, useState } from "react";
import { ChevronDown } from "lucide-react";
import { motion } from "framer-motion"; // Import framer-motion
import hero from "../assets/hero.png"; // Import your image here
import { Link } from "react-scroll"; // Import Link from react-scroll

const Hero = () => {
  // State to store the numbers
  const [doctorCount, setDoctorCount] = useState(0);
  const [hospitalCount, setHospitalCount] = useState(0);
  const [accuracyCount, setAccuracyCount] = useState(0);

  // Function to animate numbers
  const countUp = (start, end, setter) => {
    let current = start;
    const step = end / 100; // Divide by 100 to smoothen the increment
    const interval = setInterval(() => {
      current += step;
      setter(Math.min(Math.floor(current), end));
      if (current >= end) {
        clearInterval(interval);
      }
    }, 15); // duration of 15ms
  };

  // Trigger the number counting when the component mounts
  useEffect(() => {
    countUp(0, 400, setDoctorCount);
    countUp(0, 85, setHospitalCount);
    countUp(0, 100, setAccuracyCount);
  }, []);

  return (
    <section className="h-screen flex items-center -mt-8 bg-gray-50 overflow-x-hidden">
      <div className="max-w-7xl mx-auto px-4 w-full flex flex-col-reverse md:flex-row md:items-center md:gap-12 gap-4 md:space-y-0">
        {/* Left Column - Text Content */}
        <div className="md:w-1/2 text-center md:text-left md:pt-20">
          <h1
            className="text-3xl md:text-6xl font-bold text-gray-900 mb-6 leading-tight"
            data-aos="fade-right"
            data-aos-delay="300"
          >
            AI-Powered Eye Health Analysis
          </h1>
          <p
            className="md:text-xl text-[16px] text-gray-600 mb-8"
            data-aos="fade-right"
            data-aos-delay="300"
          >
            Revolutionizing eye care with advanced OCT scan and Fundus Images
            analysis, AI-driven prescription assistance, and real-time health
            recommendations through cutting-edge deep learning technology.
          </p>

          <Link to="analysis-section" smooth={true} duration={300}>
            <button
              className="bg-blue-600 text-white md:px-8 px-6 md:py-3 py-2.5 rounded-lg hover:bg-blue-700 transition mb-12 flex items-center border-2 border-transparent hover:border-blue-300 mx-auto md:mx-0"
              id="clrbtn"
            >
              Get Started
              <ChevronDown className="ml-2" size={20} />
            </button>
          </Link>

          {/* Metrics */}
          <div className="grid grid-cols-3 gap-8">
            <div>
              <motion.div
                className="md:text-3xl text-2xl font-bold text-gray-600 mb-2"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 1 }}
              >
                <motion.span key={doctorCount} className="font-bold">
                  {doctorCount}+
                </motion.span>
              </motion.div>
              <div className="text-gray-600 md:text-base text-xs">
                Doctors Trust Us
              </div>
            </div>
            <div>
              <motion.div
                className="md:text-3xl text-2xl font-bold text-gray-600 mb-2"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 1 }}
              >
                <motion.span key={hospitalCount} className="font-bold">
                  {hospitalCount}+
                </motion.span>
              </motion.div>
              <div className="text-gray-600 md:text-base text-xs">
                Across Hospitals
              </div>
            </div>
            <div>
              <motion.div
                className="md:text-3xl text-2xl font-bold text-gray-600 mb-2"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 1 }}
              >
                <motion.span key={accuracyCount} className="font-bold">
                  {accuracyCount}%
                </motion.span>
              </motion.div>
              <div className="text-gray-600 md:text-base text-xs">
                Accurate up to
              </div>
            </div>
          </div>
        </div>
        {/* Right Column - Image */}
        <div
          className="md:w-1/2 flex justify-center"
          data-aos="fade-left"
          data-aos-delay="300"
        >
          <img src={hero} alt="Hero" className="w-full h-auto max-w-lg" />
        </div>
      </div>
    </section>
  );
};

export default Hero;
