import React from 'react'

const Chatbot = () => {
  return (
    <div className='overflow-x-hidden'>
         {/* Chatbot Section */}
      <section className="py-16 bg-gray-50">
        <div className="max-w-3xl mx-auto px-4">
          <h2 className="text-3xl font-bold text-center mb-12" data-aos="fade-right" data-aos-delay="300">
            Eye Health Assistant
          </h2>
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="h-96 border border-gray-200 rounded-lg mb-4 p-4 overflow-y-auto">
              <div className="mb-4">
                <div className="bg-blue-100 rounded-lg p-3 inline-block" data-aos="fade-up" data-aos-delay="300">
                  Hello! How can I help you with your eye health today?
                </div>
              </div>
            </div>
            <div className="flex gap-2">
              <input
                type="text"
                className="flex-1 p-2 border border-gray-300 rounded-md"
                placeholder="Type your message..."
              />
              <button className="bg-blue-600 text-white px-6 py-2 rounded-md hover:bg-blue-700 transition">
                Send
              </button>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

export default Chatbot