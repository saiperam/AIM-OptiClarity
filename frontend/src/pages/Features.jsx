import React from "react";
import { Eye, Stethoscope, MessageCircle } from "lucide-react";
export const Features = () => {
  return (
    <div className="w-full min-h-screen bg-white overflow-x-hidden">
      <div className="max-w-7xl mx-auto px-4 py-16">
        <h1 className="sm:text-4xl text-3xl font-bold text-center mb-16" data-aos="fade-top" data-aos-delay="300">Our Features</h1>
        <div className="grid md:grid-cols-3 gap-8">
          {/* OCT & Fundus Analysis */}
          <div className="bg-white p-8 rounded-xl shadow-lg hover:shadow-xl transition" data-aos="fade-right" data-aos-delay="300">
            <div className="bg-blue-100 w-16 h-16 rounded-lg flex items-center justify-center mb-6">
              <Eye className="text-blue-600" size={32} />
            </div>
            <h3 className="text-xl font-semibold mb-4">Eye Scan Analysis</h3>
            <p className="text-gray-600 mb-4">
              Advanced AI analysis of OCT scans and Fundus images for accurate
              disease classification using state-of-the-art deep learning
              models.
            </p>
            <ul className="text-gray-600 space-y-2">
              <li>• Rapid disease detection</li>
              <li>• High accuracy results</li>
              <li>• Multiple scan support</li>
            </ul>
          </div>
          {/* Prescription Assistance */}
          <div className="bg-white p-8 rounded-xl shadow-lg hover:shadow-xl transition" data-aos="fade-top" data-aos-delay="300">
            <div className="bg-green-100 w-16 h-16 rounded-lg flex items-center justify-center mb-6">
              <Stethoscope className="text-green-600" size={32} />
            </div>
            <h3 className="text-xl font-semibold mb-4">
              Prescription Assistant
            </h3>
            <p className="text-gray-600 mb-4">
              AI-powered prescription suggestions based on comprehensive
              analysis of refractive error data and patient history.
            </p>
            <ul className="text-gray-600 space-y-2">
              <li>• Personalized recommendations</li>
              <li>• Data-driven analysis</li>
              <li>• Historical tracking</li>
            </ul>
          </div>
          {/* Health Care Chatbot */}
          <div className="bg-white p-8 rounded-xl shadow-lg hover:shadow-xl transition" data-aos="fade-left" data-aos-delay="300">
            <div className="bg-purple-100 w-16 h-16 rounded-lg flex items-center justify-center mb-6">
              <MessageCircle className="text-purple-600" size={32} />
            </div>
            <h3 className="text-xl font-semibold mb-4">AI Health Assistant</h3>
            <p className="text-gray-600 mb-4">
              Interactive chatbot providing personalized eye health care
              suggestions and recommendations.
            </p>
            <ul className="text-gray-600 space-y-2">
              <li>• 24/7 availability</li>
              <li>• Expert knowledge base</li>
              <li>• Personalized guidance</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
