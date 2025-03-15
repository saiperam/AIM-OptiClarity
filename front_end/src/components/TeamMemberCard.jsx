import React from "react";
import { motion } from "framer-motion";

export const TeamMemberCard = ({ name, role, image }) => {
  return (
    <div className="flex flex-col items-center">
      <div className="relative w-[280px] h-[280px] mb-6">
        <motion.div
          initial={{
            opacity: 0,
          }}
          animate={{
            opacity: 1,
            transition: {
              delay: 0.2,
              duration: 0.4,
              ease: "easeIn",
            },
          }}
        >
          <motion.div
            initial={{
              opacity: 0,
            }}
            animate={{
              opacity: 1,
              transition: {
                delay: 0.2,
                duration: 0.4,
                ease: "easeIn",
              },
            }}
            className="w-[298px] h-[298px] rounded-full absolute ml-[2px] mt-[2px]"
          >
            <img
              src={image}
              alt={name}
              className="w-[280px] h-[280px] object-cover rounded-full"
            />
          </motion.div>
          <motion.svg
            className="w-[285px] h-[285px]"
            fill="transparent"
            viewBox="0 0 506 506"
            xmlns="http://www.w3.org/2000/svg"
          >
            {/* Outer circle */}
            <motion.circle
              cx="253"
              cy="253"
              r="251"
              stroke="currentColor"
              className="text-gray-700"
              strokeWidth="4"
              strokeLinecap="round"
              strokeLinejoin="round"
              initial={{
                strokeDasharray: "24 10 0 0",
              }}
              animate={{
                strokeDasharray: ["15 120 25 25", "16 25 92 72", "4 250 22 22"],
                rotate: [120, 360],
              }}
              transition={{
                duration: 20,
                repeat: Infinity,
                repeatType: "reverse",
              }}
            />
          </motion.svg>
        </motion.div>
      </div>
      <motion.h3
        initial={{
          opacity: 0,
          y: 10,
        }}
        animate={{
          opacity: 1,
          y: 0,
        }}
        transition={{
          delay: 0.3,
          duration: 0.4,
        }}
        className="text-xl font-semibold mb-2 text-gray-900 dark:text-white text-center"
      >
        {name}
      </motion.h3>
      <motion.p
        initial={{
          opacity: 0,
          y: 10,
        }}
        animate={{
          opacity: 1,
          y: 0,
        }}
        transition={{
          delay: 0.4,
          duration: 0.4,
        }}
        className="text-gray-600 dark:text-gray-400 text-center"
      >
        {role}
      </motion.p>
    </div>
  );
};