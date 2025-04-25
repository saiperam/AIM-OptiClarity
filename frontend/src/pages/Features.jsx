import React from "react";
import { Eye, CalendarCheck, MessageCircle } from "lucide-react";
export const Features = () => {
  return (
    <div className="w-full min-h-screen bg-white overflow-x-hidden">
      <div className="max-w-7xl mx-auto px-4 py-16">
        <h1
          className="sm:text-4xl text-3xl font-bold text-center mb-16"
          data-aos="fade-top"
          data-aos-delay="300"
        >
          Our Features
        </h1>
        <div className="grid md:grid-cols-3 gap-8">
          {/* OCT & Fundus Analysis */}
          <div
            className="bg-white p-8 rounded-xl shadow-lg hover:shadow-xl transition"
            data-aos="fade-right"
            data-aos-delay="300"
          >
            <div className="bg-blue-100 w-16 h-16 rounded-lg flex items-center justify-center mb-6">
              <Eye className="text-blue-600" size={32} />
            </div>
            <h3 className="text-xl font-semibold mb-4">Eye Scan Analysis</h3>
            <p className="text-gray-600 mb-4">
              Advanced AI analysis of four different scan types including OCT,
              Fundus, Slit Lamp, and Corneal Topography for accurate disease
              classification using state-of-the-art deep learning models.
            </p>
            <ul className="text-gray-600 space-y-2">
              <li>• Rapid disease detection</li>
              <li>• High accuracy results</li>
              <li>• Multiple scan support</li>
            </ul>
          </div>

          {/* Health Care Chatbot */}
          <div
            className="bg-white p-8 rounded-xl shadow-lg hover:shadow-xl transition"
            data-aos="fade-top"
            data-aos-delay="300"
          >
            <div className="bg-purple-100 w-16 h-16 rounded-lg flex items-center justify-center mb-6">
              <MessageCircle className="text-purple-600" size={32} />
            </div>
            <h3 className="text-xl font-semibold mb-4">AI Health Assistant</h3>
            <p className="text-gray-600 mb-4">
            AI-powered assistant delivering intelligent support for eye condition assessment and clinical decision-making.
            </p>
            <ul className="text-gray-600 space-y-2">
              <li>• 24/7 diagnostic assistance</li>
              <li>• Built on expert ophthalmic knowledge</li>
              <li>• Context-aware clinical insights</li>
            </ul>
          </div>

          {/* Prescription Assistance */}
          <div
            className="bg-white p-8 rounded-xl shadow-lg hover:shadow-xl transition"
            data-aos="fade-left"
            data-aos-delay="300"
          >
            <div className="bg-green-100 w-16 h-16 rounded-lg flex items-center justify-center mb-6">
              <CalendarCheck className="text-green-600" size={32} />
            </div>
            <h3 className="text-xl font-semibold mb-4">
              Optometrist Scheduler
            </h3>
            <p className="text-gray-600 mb-4">
              Seamless appointment scheduling system designed for optometrists,
              allowing efficient management of patient visits, availability, and
              consultation times through an intelligent, user-friendly
              interface.
            </p>
            <ul className="text-gray-600 space-y-2">
              <li>• Smart appointment management</li>
              <li>• Conflict-free time slot booking</li>
              <li>• Customizable availability and buffer settings</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
