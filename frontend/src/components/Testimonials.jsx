import React from "react";

const Testimonials = () => {
  return (
    <div className="bg-gray-50 py-20">
      <div className="max-w-7xl mx-auto px-6 md:px-0">
        <div className="flex flex-col items-center text-center space-y-6">
          <h2 className="text-3xl font-bold text-blue-900 relative" data-aos="fade-right" data-aos-delay="100">
            Trusted by Eye-health Professionals
            <span className="block w-16 h-1 bg-gradient-to-r from-blue-300 to-blue-900 mx-auto mt-4 rounded-full"></span>
          </h2>
          <p className="text-gray-600 sm:text-lg text-base text-center">
            See what medical experts are saying about our platform.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-12 px-4 md:px-0" data-aos="fade-top" data-aos-delay="300"> 
          <Testimonial
            text="As an ophthalmologist, I'm amazed by how this platform has enhanced our diagnostic capabilities. The accuracy and speed are remarkable."
            author="Dr. Aisha Patel"
            role="Ophthalmologist"
          />
          <Testimonial
            text="This system has transformed our workflow. We can now provide faster and more accurate diagnoses to our patients."
            author="Dr. Benjamin Lee"
            role="Optometrist"
          />
          <Testimonial
            text="The intuitive interface and reliable results make this an essential tool in our daily practice. Highly recommended."
            author="Dr. Sophia Martinez"
            role="Retina Specialist"
          />
        </div>
      </div>
    </div>
  );
};

const Testimonial = ({ text, author, role }) => {
  return (
    <div className="bg-white shadow-lg rounded-2xl p-6 text-center">
      <p className="text-gray-700 italic">"{text}"</p>
      <h4 className="mt-4 font-semibold text-lg text-gray-900">{author}</h4>
      <p className="text-sm text-gray-500">{role}</p>
    </div>
  );
};

export default Testimonials;
