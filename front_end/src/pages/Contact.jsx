import React from "react";
import { Github, Linkedin, Mail, Twitter } from "lucide-react";

export const Contact = () => {
  return (
    <div className="w-full min-h-screen bg-white overflow-x-hidden">
      <div className="max-w-7xl mx-auto px-4 py-16">
        <h1 className="text-4xl font-bold text-center mb-6" data-aos="fade-top" data-aos-delay="300">Get in Touch</h1>
        {/* Social Links */}
        <div className="flex justify-center space-x-8 mb-4">
          <a
            href="https://linkedin.com"
            className="p-3 bg-gray-100 rounded-full hover:bg-gray-200 transition"
          >
            <Linkedin className="text-gray-700" size={24} />
          </a>
          <a
            href="https://twitter.com"
            className="p-3 bg-gray-100 rounded-full hover:bg-gray-200 transition"
          >
            <Twitter className="text-gray-700" size={24} />
          </a>
          <a
            href="mailto:contact@opticlarity.com"
            className="p-3 bg-gray-100 rounded-full hover:bg-gray-200 transition"
          >
            <Mail className="text-gray-700" size={24} />
          </a>
        </div>
        {/* Contact Form */}
        <div className="max-w-2xl mx-auto">
          <form className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Name
              </label>
              <input
                type="text"
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-black focus:border-transparent"
                placeholder="Your name"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Email
              </label>
              <input
                type="email"
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-black focus:border-transparent"
                placeholder="your@email.com"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Message
              </label>
              <textarea
                rows={4}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-black focus:border-transparent"
                placeholder="Your message..."
              />
            </div>
            <button
              type="submit"
              className="w-full bg-blue-600 text-white py-3 rounded-lg hover:bg-blue-700 transition"
              id="clrbtn"
            >
              Send Message
            </button>
          </form>
        </div>
      </div>
    </div>
  );
};
