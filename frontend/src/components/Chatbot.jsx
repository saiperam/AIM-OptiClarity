import React, { useState } from "react";

const Chatbot = () => {
  const [messages, setMessages] = useState([
    {
      text: "Hello! How can I help you with your eye health today?",
      sender: "bot",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { text: input, sender: "user" };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: input }), // Changed from prompt to query to match backend optibot.py
      });

      const data = await response.json();
      if (response.ok) {
        const botMessage = {
          text: data.response || "Sorry, I didn't understand that.",
          sender: "bot",
        };
        setMessages((prevMessages) => [...prevMessages, botMessage]);
      } else {
        const botMessage = {
          text: data.error || "Error processing your request.",
          sender: "bot",
        };
        setMessages((prevMessages) => [...prevMessages, botMessage]);
      }
    } catch (error) {
      console.error("Error fetching response:", error);
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: "Error processing your request.", sender: "bot" },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="overflow-x-hidden">
      <section className="py-16 bg-gray-50">
        <div className="max-w-3xl mx-auto px-4">
          <h2 className="text-3xl font-bold text-center text-blue-900">
            Your Eye Health Advisor
          </h2>
          <span className="block w-16 h-1 bg-gradient-to-r from-blue-300 to-blue-900 mx-auto mt-4 mb-6 rounded-full"></span>
          <p className="text-gray-600 sm:text-lg text-base text-center mb-12">
            Get instant eye health insights and recommendations from our intelligent chatbot.
          </p>
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="h-96 border border-gray-200 rounded-lg mb-4 p-4 overflow-y-auto">
              {messages.map((msg, index) => (
                <div
                  key={index}
                  className={`mb-4 ${
                    msg.sender === "bot" ? "text-left" : "text-right"
                  }`}
                >
                  <div
                    className={`inline-block p-3 rounded-lg ${
                      msg.sender === "bot"
                        ? "bg-blue-100"
                        : "bg-blue-600 text-white"
                    }`}
                  >
                    {msg.text}
                  </div>
                </div>
              ))}
            </div>
            <div className="flex gap-2">
              <input
                type="text"
                className="flex-1 p-2 border border-gray-300 rounded-md"
                placeholder="Type your message..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && sendMessage()}
                disabled={loading}
              />
              <button
                className="bg-blue-800 text-white px-6 py-2 rounded-md hover:bg-blue-700 transition"
                onClick={sendMessage}
                disabled={loading}
              >
                {loading ? "Sending..." : "Send"}
              </button>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Chatbot;













































{/*import React, { useState } from "react";

const Chatbot = () => {
  const [messages, setMessages] = useState([
    {
      text: "Hello! How can I help you with your eye health today?",
      sender: "bot",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { text: input, sender: "user" };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch("http://localhost:8000/generate_text", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ prompt: input, max_length: 200 }), // Match backend max_length
      });

      const data = await response.json();
      if (response.ok) {
        const botMessage = {
          text: data.text || "Sorry, I didn't understand that.",
          sender: "bot",
        };
        setMessages((prevMessages) => [...prevMessages, botMessage]);
      } else {
        const botMessage = {
          text: data.error || "Error processing your request.",
          sender: "bot",
        };
        setMessages((prevMessages) => [...prevMessages, botMessage]);
      }
    } catch (error) {
      console.error("Error fetching response:", error);
      setMessages((prevMessages) => [
        ...prevMessages,
        { text: "Error processing your request.", sender: "bot" },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="overflow-x-hidden">
      <section className="py-16 bg-gray-50">
        <div className="max-w-3xl mx-auto px-4">
          <h2
            className="text-3xl font-bold text-center text-blue-900"
            data-aos="fade-left"
            data-aos-delay="300"
          >
            Your Eye Health Advisor
          </h2>
          <span className="block w-16 h-1 bg-gradient-to-r from-blue-300 to-blue-900 mx-auto mt-4 mb-6 rounded-full"></span>
          <p className="text-gray-600 sm:text-lg text-base text-center mb-12">
            Get instant eye health insights and recommendations
            from our intelligent chatbot.
          </p>
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="h-96 border border-gray-200 rounded-lg mb-4 p-4 overflow-y-auto">
              {messages.map((msg, index) => (
                <div
                  key={index}
                  className={`mb-4 ${
                    msg.sender === "bot" ? "text-left" : "text-right"
                  }`}
                >
                  <div
                    className={`inline-block p-3 rounded-lg ${
                      msg.sender === "bot"
                        ? "bg-blue-100"
                        : "bg-blue-600 text-white"
                    }`}
                  >
                    {msg.text}
                  </div>
                </div>
              ))}
            </div>
            <div className="flex gap-2">
              <input
                type="text"
                className="flex-1 p-2 border border-gray-300 rounded-md"
                placeholder="Type your message..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && sendMessage()}
                disabled={loading}
              />
              <button
                className="bg-blue-800 text-white px-6 py-2 rounded-md hover:bg-blue-700 transition"
                onClick={sendMessage}
                disabled={loading}
              >
                {loading ? "Sending..." : "Send"}
              </button>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Chatbot;*/}

