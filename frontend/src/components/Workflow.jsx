import React, { useState } from 'react';
import {
  UploadIcon,
  ImageIcon,
  BrainCircuitIcon,
  BarChart3Icon,
} from 'lucide-react';

export default function Workflow() {
  const [activeStep, setActiveStep] = useState(null);

  const steps = [
    {
      id: 1,
      title: 'Upload Image Scan',
      description: 'Select and upload medical image scan for analysis',
      icon: <UploadIcon className="h-6 w-6 sm:h-8 sm:w-8" />, // Adjusted size for responsiveness
    },
    {
      id: 2,
      title: 'Preprocessing',
      description: 'Normalize, resize, and enhance image features',
      icon: <ImageIcon className="h-6 w-6 sm:h-8 sm:w-8" />, // Adjusted size for responsiveness
    },
    {
      id: 3,
      title: 'Deep Learning Model',
      description: 'Analyze image using trained neural network',
      icon: <BrainCircuitIcon className="h-6 w-6 sm:h-8 sm:w-8" />, // Adjusted size for responsiveness
    },
    {
      id: 4,
      title: 'Classification Results',
      description: 'View detected condition and confidence score',
      icon: <BarChart3Icon className="h-6 w-6 sm:h-8 sm:w-8" />, // Adjusted size for responsiveness
    },
  ];

  return (
    <div className="max-w-5xl mx-auto px-4 sm:px-8 py-8 sm:py-16"> {/* Adjusted padding for smaller screens */}
      <div className="bg-white rounded-lg shadow-lg p-4 sm:p-8" data-aos="fade-top" data-aos-delay="300"> {/* Adjusted padding for responsiveness */}
        <div className="flex flex-col sm:flex-row justify-between items-center sm:items-stretch"> {/* Adjusted layout for smaller screens */}
          {steps.map((step, index) => (
            <div key={step.id} className="flex flex-col items-center relative flex-1 mb-6 sm:mb-0"> {/* Added spacing for mobile */}
              <button
                onClick={() => setActiveStep(step.id)}
                className={`w-10 h-10 sm:w-12 sm:h-12 rounded-full flex items-center justify-center z-10 transition-colors
                  ${step.id === activeStep ? 'bg-blue-600' : 'bg-gray-200 hover:bg-gray-300'}`}
              >
                <div className={step.id === activeStep ? 'text-white' : 'text-gray-500'}>
                  {step.icon}
                </div>
              </button>
              <div className="mt-2 sm:mt-3 text-center">
                <p className={`text-sm sm:text-base font-medium ${step.id === activeStep ? 'text-blue-600' : 'text-gray-500'}`}>{step.title}</p>
                <p className="text-xs sm:text-sm text-gray-500 mt-1 max-w-[130px] sm:max-w-[150px]">{step.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
