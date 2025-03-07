import { useState, useRef } from 'react';
import axios from 'axios';
import { FlipWords } from "./ui/flip-words";

interface WeatherData {
    weather: { main: string; description: string; icon: string }[];
    main :{ temp : number , humidity : number , feels_like: number ,pressure:string ,  }
    visibility : string 
  }

  const openweatherKey = '763e13cdd2f2a1ce54c868f24f7a0cec';

function PestDetection() {
    const [selectedFile, setSelectedFile] = useState(null);
    const [preview, setPreview] = useState(null);
    const [result, setResult] = useState('');
    const [showCamera, setShowCamera] = useState(false);
    const [forecast, setForecast] = useState<WeatherData | null>(null);
    
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    
    const words = ["Detection", "Prevention"];
    
    const handleFileChange = (event) => {
        const file = event.target.files[0];
        setSelectedFile(file);
        setPreview(URL.createObjectURL(file));  // Show preview immediately
    };

    const handleCapture = async () => {
        setShowCamera(true);
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                videoRef.current.play();
            }
        } catch (error) {
            console.error('Error accessing camera', error);
            alert('Could not access camera');
        }
    };
    
    const captureImage = () => {
        if (!videoRef.current || !canvasRef.current) return;
        
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        canvas.toBlob((blob) => {
            const file = new File([blob], 'captured_image.png', { type: 'image/png' });
            setSelectedFile(file);
            setPreview(URL.createObjectURL(file)); // Show captured preview
            setShowCamera(false);
        }, 'image/png');
        
        const stream = video.srcObject;
        stream?.getTracks().forEach(track => track.stop());
        video.srcObject = null;
    };
    
    const handleUpload = async () => {
        if (!selectedFile) {
            alert('Please select or capture an image first!');
            return;
        }
    
        // First, get location coordinates
        navigator.geolocation.getCurrentPosition(async (position) => {
            const lat = position.coords.latitude;
            const lon = position.coords.longitude;
    
            try {
                // Fetch weather based on location
                const weatherResponse = await axios.get(
                    `https://api.openweathermap.org/data/2.5/weather?lat=${lat}&lon=${lon}&units=metric&appid=${openweatherKey}`
                );
    
                const forecast = weatherResponse.data;
    
                // Extract required weather data
                const temperature = forecast?.main?.temp;
                const humidity = forecast?.main?.humidity;
                const weatherMain = forecast?.weather?.[0]?.main;
    
                // Now, upload image + weather data together
                const formData = new FormData();
                formData.append('image', selectedFile);
                formData.append('temperature', temperature);
                formData.append('humidity', humidity);
                formData.append('weather_main', weatherMain);
    
                const response = await axios.post('http://127.0.0.1:8000/analyze_image', formData, {
                    headers: { 'Content-Type': 'multipart/form-data' },
                });
    
                const { predicted_disease, prevention_measures } = response.data;
                setResult(`Detected: ${predicted_disease}\nPrevention: ${prevention_measures}`);
            } catch (error) {
                console.error('Upload or weather fetch failed', error);
                setResult('Failed to analyze image.');
            }
        }, (error) => {
            console.error('Geolocation error:', error);
            alert('Failed to get location. Please allow location access and try again.');
        });
    };
    
    
    const clearImage = () => {
        setSelectedFile(null);
        setPreview(null);
        setResult('');
    };


    return (
        <div className='p-10 mx-10 my-2 flex gap-10 items-center'>
            {/* Preview Box */}
            <div className="mt-4 w-[30vw] h-[70vh] border rounded-2xl flex items-center justify-center bg-gray-100">
                {preview ? (
                    <img src={preview} alt="Preview" className="max-w-full max-h-full object-contain" />
                ) : (
                    <span className="text-gray-500">No image selected</span>
                )}
            </div>

            {/* Right Section */}
            <div className='w-[50vw] h-[70vh]'>
                <div className="justify-center items-center">
                    <div className="text-7xl mx-2 text-neutral-600 dark:text-black font-bold">
                        Pest
                        <FlipWords words={words} /> <br />
                    </div>
                </div>

                <div className='p-4 mt-3'>
                    {/* File Upload */}
                    <label className="custom-upload-button bg-green-500 text-white px-4 py-2 rounded cursor-pointer">
                        Select Image
                        <input
                            type="file"
                            accept="image/*"
                            onChange={handleFileChange}
                            style={{ display: 'none' }}
                        />
                    </label>

                    {/* Open Camera Button */}
                    <button className="ml-2 bg-blue-500 text-white px-4 py-2 rounded" onClick={handleCapture}>
                        Open Camera & Capture Image
                    </button>

                    {/* Clear Image Button */}
                    <button 
                        className="ml-2 bg-red-500 text-white px-4 py-2 rounded" 
                        onClick={clearImage} 
                        disabled={!preview} // disable if no image
                    >
                        Clear Image
                    </button>

                    {/* Camera Capture UI */}
                    {showCamera && (
                        <div className="mt-4">
                            <video ref={videoRef} autoPlay className="w-full max-w-md border rounded"></video>
                            <button
                                className="mt-2 bg-purple-500 text-white px-4 py-2 rounded"
                                onClick={captureImage}
                            >
                                Capture Image
                            </button>
                        </div>
                    )}

                    <canvas ref={canvasRef} style={{ display: 'none' }}></canvas>

                    {/* Upload Button */}
                    <div className="mt-4">
                        <button className="bg-orange-500 text-white px-4 py-2 rounded" onClick={handleUpload}>
                            Upload and Analyze
                        </button>
                    </div>

                    {/* Analysis Result */}
                    {result && (
                        <div className="mt-4 p-2 bg-green-100 border border-green-500 rounded">
                            <strong>Result:</strong>
                            <pre className="whitespace-pre-line">{result}</pre>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

export default PestDetection;


