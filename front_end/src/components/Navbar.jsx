import React, { useState } from "react";
import { Menu, X, Home, Brain, Info, Mail } from "lucide-react";
import { NavLink } from "react-router-dom";
import navlogo from "../assets/navlogo.png";

export const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <nav className="w-full bg-white shadow-sm z-50">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center">
            <NavLink to="/">
              <img src={navlogo} alt="Logo" className="h-10 -ml-2 md:ml-0 md:h-14 w-auto" />
            </NavLink>
          </div>
          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            <NavLink
              to="/"
              end
              className={({ isActive }) =>
                `flex items-center space-x-1 hover:text-black ${
                  isActive ? "text-black" : "text-gray-700"
                }`
              }
            >
              <Home size={18} />
              <span>Home</span>
            </NavLink>
            <NavLink
              to="/features"
              className={({ isActive }) =>
                `flex items-center space-x-1 hover:text-black ${
                  isActive ? "text-black" : "text-gray-700"
                }`
              }
            >
              <Brain size={18} />
              <span>Features</span>
            </NavLink>
            <NavLink
              to="/about"
              className={({ isActive }) =>
                `flex items-center space-x-1 hover:text-black ${
                  isActive ? "text-black" : "text-gray-700"
                }`
              }
            >
              <Info size={18} />
              <span>About</span>
            </NavLink>
            <NavLink
              to="/contact"
              className={({ isActive }) =>
                `flex items-center space-x-1 hover:text-black ${
                  isActive ? "text-black" : "text-gray-700"
                }`
              }
            >
              <Mail size={18} />
              <span>Contact</span>
            </NavLink>
          </div>
          {/* Mobile Menu Button */}
          <div className="md:hidden flex items-center">
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="text-gray-600 hover:text-black"
            >
              {isOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>
      </div>
      {/* Mobile Navigation */}
      {isOpen && (
        <div className="md:hidden pb-4 px-4 bg-white shadow-md absolute right-0 left-0 z-40">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              `block py-2 hover:text-black ${
                isActive ? "text-black" : "text-gray-700"
              }`
            }
            onClick={() => setIsOpen(false)}
          >
            Home
          </NavLink>
          <NavLink
            to="/features"
            className={({ isActive }) =>
              `block py-2 hover:text-black ${
                isActive ? "text-black" : "text-gray-700"
              }`
            }
            onClick={() => setIsOpen(false)}
          >
            Features
          </NavLink>
          <NavLink
            to="/about"
            className={({ isActive }) =>
              `block py-2 hover:text-black ${
                isActive ? "text-black" : "text-gray-700"
              }`
            }
            onClick={() => setIsOpen(false)}
          >
            About
          </NavLink>
          <NavLink
            to="/contact"
            className={({ isActive }) =>
              `block py-2 hover:text-black ${
                isActive ? "text-black" : "text-gray-700"
              }`
            }
            onClick={() => setIsOpen(false)}
          >
            Contact
          </NavLink>
        </div>
      )}
    </nav>
  );
};
