import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { Upload, X } from "lucide-react";

const UploadScans = () => {
    const [selectedFile, setSelectedFile] = useState();
    const [preview, setPreview] = useState();
    const [data, setData] = useState();
    const [isLoading, setIsLoading] = useState(false);
    const [confidence, setConfidence] = useState(0);
    const [showPopup, setShowPopup] = useState(false);
    
    const fileInputRef = useRef(null);
  
    const sendFile = async () => {
      if (selectedFile) {
        let formData = new FormData();
        formData.append("file", selectedFile);
        setIsLoading(true);
        let res = await axios.post("http://localhost:8002/predict", formData);
        if (res.status === 200) {
          setData(res.data);
          setConfidence((parseFloat(res.data.confidence) * 100).toFixed(2));
          setShowPopup(true);
        }
        setIsLoading(false);
      }
    };
  
    const clearData = () => {
      setData(null);
      setSelectedFile(null);
      setPreview(null);
      setConfidence(0);
      setShowPopup(false);
    };
  
    useEffect(() => {
      if (!selectedFile) {
        setPreview(undefined);
        return;
      }
      const objectUrl = URL.createObjectURL(selectedFile);
      setPreview(objectUrl);
    }, [selectedFile]);
  
    useEffect(() => {
      if (!preview) return;
      sendFile();
    }, [preview]);
  
    const onSelectFile = (files) => {
      if (!files || files.length === 0) {
        setSelectedFile(undefined);
        setData(undefined);
        return;
      }
      setSelectedFile(files[0]);
      setData(undefined);
    };
  
    return (
      <section className="py-16 bg-gray-50 overflow-x-hidden" id="analysis-section">
        <div className="max-w-7xl mx-auto px-4">
          <h2 className="text-3xl font-bold text-center mb-12" data-aos="fade-right" data-aos-delay="300">Image Analysis</h2>
          <div className="grid md:grid-cols-2 gap-8">
            {/* OCT Scan Upload */}
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-500 transition">
              <Upload className="mx-auto mb-4 text-gray-400" size={48} />
              <h3 className="text-xl font-semibold mb-2">Upload OCT Scan</h3>
              <p className="text-gray-600 mb-4">Drag and drop your OCT scan or click to browse</p>
              <button
                onClick={() => fileInputRef.current.click()}
                className="bg-white border border-blue-600 text-blue-600 px-6 py-2 rounded hover:bg-blue-50 transition"
              >
                Select File
              </button>
            </div>
            {/* Fundus Image Upload */}
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-500 transition">
              <Upload className="mx-auto mb-4 text-gray-400" size={48} />
              <h3 className="text-xl font-semibold mb-2">Upload Fundus Image</h3>
              <p className="text-gray-600 mb-4">Drag and drop your fundus image or click to browse</p>
              <button
                onClick={() => fileInputRef.current.click()}
                className="bg-white border border-blue-600 text-blue-600 px-6 py-2 rounded hover:bg-blue-50 transition"
              >
                Select File
              </button>
            </div>
          </div>
        </div>
        
        <input
          ref={fileInputRef}
          type="file"
          accept=".jpg,.jpeg,.png,.dicom"
          onChange={(e) => onSelectFile(e.target.files)}
          className="hidden"
        />
      
        {/* Popup Modal */}
        {showPopup && (
          <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
            <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg md:w-96 w-80 relative">
              <button onClick={clearData} className="absolute top-2 right-2 text-gray-700 p-2">
                <X size={20} />
              </button>
              {isLoading ? (
                <div className="text-center text-gray-600">Processing...</div>
              ) : (
                <div className="text-center">
                  <img src={preview} alt="Uploaded scan" className="md:w-80 md:h-80 w-60 h-60 object-cover rounded-lg mx-auto mb-4" />
                  <div className="text-lg font-semibold">{data?.class}</div>
                  <div className="text-gray-500">Confidence: {confidence}%</div>
                </div>
              )}
            </div>
          </div>
        )}
      </section>
    );
};

export default UploadScans;
