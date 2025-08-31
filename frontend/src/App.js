import React, { useState, useEffect } from 'react';
import Papa from 'papaparse';
import Dashboard from './components/Dashboard';
import HashtagsTrending from './components/HashtagsTrending';
import './App.css';

function App() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('dashboard');

  useEffect(() => {
    // Load CSV data
    const loadCSVData = async () => {
      try {
        const response = await fetch('/nodes_export_for_frontend (1).csv');
        const csvText = await response.text();
        
        Papa.parse(csvText, {
          header: true,
          complete: (results) => {
            // Filter out the "Unnamed: 0" entry and clean the data
            const cleanData = results.data.filter(row => 
              row.Label && row.Label !== 'Unnamed: 0' && 
              row['Closeness Centrality'] !== '0.0'
            ).map(row => ({
              ...row,
              'Closeness Centrality': parseFloat(row['Closeness Centrality']),
              'Betweenness Centrality': parseFloat(row['Betweenness Centrality']),
              'Eccentricity': parseInt(row['Eccentricity']),
              'Modularity Class': parseInt(row['Modularity Class'])
            }));
            
            setData(cleanData);
            setLoading(false);
          },
          error: (error) => {
            console.error('Error parsing CSV:', error);
            setLoading(false);
          }
        });
      } catch (error) {
        console.error('Error loading CSV:', error);
        setLoading(false);
      }
    };

    loadCSVData();
  }, []);

  const renderTabContent = () => {
    switch (activeTab) {
      case 'hashtags':
        return <HashtagsTrending />;
      case 'dashboard':
      default:
        return <Dashboard data={data} />;
    }
  };

  if (loading && activeTab === 'dashboard') {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>Loading OSINT Intelligence Dashboard...</p>
      </div>
    );
  }

  return (
    <div className="App">
      <div className="tab-navigation">
        <button 
          className={`tab-button ${activeTab === 'dashboard' ? 'active' : ''}`}
          onClick={() => setActiveTab('dashboard')}
        >
          🛡️ OSINT Dashboard
        </button>
        <button 
          className={`tab-button ${activeTab === 'hashtags' ? 'active' : ''}`}
          onClick={() => setActiveTab('hashtags')}
        >
          🔥 Live Hashtags Trending
        </button>
      </div>
      
      {renderTabContent()}
    </div>
  );
}

export default App; 