import React from "react";
import { TeamMemberCard } from "../components/TeamMemberCard";
import { imgassets } from "../assets/imgassets";
import { techstack } from "../assets/techstack";
import { Github, Instagram, Linkedin, GraduationCap } from "lucide-react";

export const About = () => {
  const team = [
    {
      name: "Sai Peram",
      role: "AIM Team Lead",
      image: imgassets.sai,
      instagram: "https://instagram.com/sai_peram",
      linkedin: "https://linkedin.com/in/sai-peram",
      github: "https://github.com/sai-peram",
    },
    {
      name: "Kunju Menon",
      role: "ML Engineer + Frontend",
      image: imgassets.kunju,
      instagram: "https://instagram.com/kunju_menon",
      linkedin: "https://linkedin.com/in/kunju-menon",
      github: "https://github.com/kunju-menon",
    },
    {
      name: "Md Abrar Al Zabir",
      role: "ML Engineer + Full Stack",
      image: imgassets.abrar,
      instagram: "https://instagram.com/abrar_zabir",
      linkedin: "https://linkedin.com/in/md-abrar-al-zabir",
      github: "https://github.com/md-abrar-al-zabir",
    },
    {
      name: "Sunay Shehaan",
      role: "ML Engineer + Backend",
      image: imgassets.sunay,
      instagram: "https://instagram.com/sunay_shehaan",
      linkedin: "https://linkedin.com/in/sunay-shehaan",
      github: "https://github.com/sunay-shehaan",
    },
    {
      name: "Joel Gurivireddy",
      role: "ML Engineer",
      image: imgassets.joel,
      instagram: "https://instagram.com/joel_gurivireddy",
      linkedin: "https://linkedin.com/in/joel-gurivireddy",
      github: "https://github.com/joel-gurivireddy",
    },
    {
      name: "Devansh Agrawal",
      role: "ML Engineer + Backend",
      image: imgassets.devansh,
      instagram: "https://instagram.com/devansh_agrawal",
      linkedin: "https://linkedin.com/in/devansh-agrawal",
      github: "https://github.com/devansh-agrawal",
    },
  ];

  
  return (
    <div className="w-full min-h-screen bg-white overflow-x-hidden">
      {/* Team Section */}
      <section className="bg-white py-10 px-4">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-4xl font-bold text-center mb-12" data-aos="fade-top" data-aos-delay="300">
            Our Team
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 sm:gap-16 gap-10 max-w-6xl mx-auto">
            {team.map((member) => (
              <div key={member.name} className="space-y-6">
                <TeamMemberCard
                  name={member.name}
                  role={member.role}
                  image={member.image}
                  data-aos="fade-top"
                  data-aos-delay="300"
                />
                <div className="text-center">
                  <div className="flex justify-center space-x-6 -mt-3">
                    <a href={member.github} target="_blank" rel="noopener noreferrer" className="text-gray-600 hover:text-black">
                      <Github size={20} />
                    </a>
                    <a href={member.instagram} target="_blank" rel="noopener noreferrer" className="text-gray-600 hover:text-black">
                      <Instagram size={20} />
                    </a>
                    <a href={member.linkedin} target="_blank" rel="noopener noreferrer" className="text-gray-600 hover:text-black">
                      <Linkedin size={20} />
                    </a>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Tech Stack Section */}
      <section className="py-16 px-4 bg-gray-50">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12" data-aos="fade-top" data-aos-delay="300">Our Tech Stack</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {techstack.map((techstack) => (
              <div key={techstack.name} className="bg-white p-6 rounded-lg shadow-sm text-center transform transition-all duration-300 hover:scale-105 hover:shadow-xl" data-aos="fade-up" data-aos-delay="300">
                <img
                  src={techstack.image}
                  alt={techstack.name}
                  className="w-16 h-16 mx-auto mb-4 rounded-full border-4 border-gray-200 shadow-md hover:shadow-2xl transition-all duration-300"
                />
                <p className="font-medium">{techstack.name}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Research Section */}
      <section className="py-16 px-4">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12" data-aos="fade-top" data-aos-delay="300">Research & References</h2>
          <div className="bg-white rounded-xl shadow-lg p-8 mb-4" data-aos="fade-right" data-aos-delay="300">
            <div className="space-y-6">
              <div className="border-l-4 border-blue-600 pl-4">
                <h4 className="font-medium">
                  Deep Learning for Diabetic Retinopathy Analysis: A Review, Research Challenges, and Future Directions
                </h4>
                <p className="text-gray-600 mt-2">
                  This comprehensive review guided our approach to implementing AI-powered retinal analysis.
                </p>
                <a
                  href="https://www.mdpi.com/1424-8220/22/18/6780"
                  target="_blank"
                  className="text-blue-600 hover:text-blue-700 inline-flex items-center mt-2"
                >
                  <GraduationCap className="mr-2" size={16} />
                  Read
                </a>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-8" data-aos="fade-left" data-aos-delay="300">
            <div className="space-y-6">
              <div className="border-l-4 border-blue-600 pl-4">
                <h4 className="font-medium">
                Deep Learning for Predicting Refractive Error From Retinal Fundus Images
                </h4>
                <p className="text-gray-600 mt-2">
                  This comprehensive review guided our approach to implementing AI-powered retinal analysis.
                </p>
                <a
                  href="https://iovs.arvojournals.org/article.aspx?articleid=2683803"
                  target="_blank"
                  className="text-blue-600 hover:text-blue-700 inline-flex items-center mt-2"
                >
                  <GraduationCap className="mr-2" size={16} />
                  Read
                </a>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};
