import React, { useState, useEffect } from 'react';
import Papa from 'papaparse';
import './HashtagsTrending.css';

const HashtagsTrending = () => {
  const [hashtagsData, setHashtagsData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadHashtagsData = async () => {
      try {
        const response = await fetch('/data_97cd739e_split_1.csv');
        const csvText = await response.text();
        
        Papa.parse(csvText, {
          header: true,
          complete: (results) => {
            // Process hashtags data
            const processedData = results.data
              .filter(row => row.hashtags && row.shares && row.likes)
              .map(row => {
                // Parse hashtags from the string format "['#hashtag1', '#hashtag2']"
                const hashtags = row.hashtags
                  .replace(/[\[\]']/g, '') // Remove brackets and quotes
                  .split(', ')
                  .filter(tag => tag.trim() !== '');
                
                return {
                  hashtags: hashtags,
                  shares: parseFloat(row.shares) || 0,
                  likes: parseInt(row.likes) || 0,
                  platform: row.platform,
                  username: row.username,
                  content: row.content_text
                };
              })
              .flatMap(row => 
                row.hashtags.map(tag => ({
                  hashtag: tag,
                  shares: row.shares,
                  likes: row.likes,
                  platform: row.platform,
                  username: row.username,
                  content: row.content
                }))
              )
              .sort((a, b) => b.shares - a.shares)
              .slice(0, 20);

            setHashtagsData(processedData);
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

    loadHashtagsData();
  }, []);

  if (loading) {
    return (
      <div className="hashtags-loading">
        <div className="loading-spinner"></div>
        <p>Loading trending hashtags...</p>
      </div>
    );
  }

  return (
    <div className="hashtags-trending">
      <div className="hashtags-header">
        <h2>🔥 Live Hashtags Trending</h2>
        <p>Top 20 hashtags by social media engagement</p>
      </div>
      
      <div className="hashtags-grid">
        {hashtagsData.map((item, index) => (
          <div key={index} className="hashtag-card">
            <div className="hashtag-rank">#{index + 1}</div>
            <div className="hashtag-content">
              <div className="hashtag-text">{item.hashtag}</div>
              <div className="hashtag-stats">
                <span className="shares-count">📤 {item.shares.toLocaleString()} shares</span>
                <span className="likes-count">❤️ {item.likes.toLocaleString()} likes</span>
              </div>
              <div className="hashtag-meta">
                <span className="platform">📱 {item.platform}</span>
                <span className="username">👤 {item.username}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
      
      {hashtagsData.length === 0 && (
        <div className="no-data">
          <p>No trending hashtags found</p>
        </div>
      )}
    </div>
  );
};

export default HashtagsTrending;
