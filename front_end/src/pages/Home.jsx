import React from 'react'
import Hero from '../components/Hero'
import UploadScans from '../components/UploadScans'
import Chatbot from '../components/Chatbot'

export const Home = () => {
  return (
    <div>
        {/* Hero Section */}
        <div>
            <Hero/>
        </div>
        {/* Upload Section */}
        <div>
            <UploadScans/>
        </div>
        {/* Chatbot Section */}
        <div>
            <Chatbot/>
        </div>    
    </div>
  )
}
