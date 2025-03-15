import React from "react";
import { Link } from "react-router-dom";
import { Linkedin, Twitter, Mail, Eye } from "lucide-react";

export const Footer = () => {
  return (
    <footer className="bg-white border-t">
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Brand Section */}
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <span className="text-xl font-bold text-black">
                OptiClarity
              </span>
            </div>
            <p className="text-gray-600 text-sm">
              Revolutionizing eye care with AI-powered analysis and assistance.
            </p>
          </div>
          {/* Quick Links */}
          <div>
            <h3 className="font-semibold mb-4">Quick Links</h3>
            <ul className="space-y-2">
              <li>
                <Link to="/" className="text-gray-600 hover:text-black">
                  Home
                </Link>
              </li>
              <li>
                <Link
                  to="/features"
                  className="text-gray-600 hover:text-black"
                >
                  Features
                </Link>
              </li>
              <li>
                <Link to="/about" className="text-gray-600 hover:text-black">
                  About
                </Link>
              </li>
              <li>
                <Link
                  to="/contact"
                  className="text-gray-600 hover:text-black"
                >
                  Contact
                </Link>
              </li>
            </ul>
          </div>
          {/* Features */}
          <div>
            <h3 className="font-semibold mb-4">Features</h3>
            <ul className="space-y-2">
              <li className="text-gray-600">Eye Scan Analysis</li>
              <li className="text-gray-600">Prescription Assistant</li>
              <li className="text-gray-600">AI Health Assistant</li>
            </ul>
          </div>
          {/* Social Links */}
          <div>
            <h3 className="font-semibold mb-4">Connect With Us</h3>
            <div className="flex space-x-4">
              <a
                href="#"
                className="text-gray-600 hover:text-black transition-colors"
              >
                <Linkedin className="w-5 h-5" />
              </a>
              <a
                href="#"
                className="text-gray-600 hover:text-black transition-colors"
              >
                <Twitter className="w-5 h-5" />
              </a>
              <a
                href="#"
                className="text-gray-600 hover:text-black transition-colors"
              >
                <Mail className="w-5 h-5" />
              </a>
            </div>
          </div>
        </div>
        <div className="border-t mt-8 pt-6">
          <p className="text-center text-gray-600 text-sm">
            © {new Date().getFullYear()} OptiClarity. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
};
