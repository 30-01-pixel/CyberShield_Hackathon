import React from 'react';
import { TrendingUp, Network, Users, Target } from 'lucide-react';
import './MetricsPanel.css';

const MetricsPanel = ({ topHubAccounts, topBridgeAccounts, peripheralAccounts, communities }) => {
  const getMetricColor = (value, maxValue, type) => {
    const percentage = value / maxValue;
    if (type === 'closeness') {
      if (percentage > 0.8) return '#10b981'; // Green for high
      if (percentage > 0.6) return '#f59e0b'; // Yellow for medium
      return '#ef4444'; // Red for low
    }
    if (type === 'betweenness') {
      if (percentage > 0.7) return '#10b981';
      if (percentage > 0.4) return '#f59e0b';
      return '#ef4444';
    }
    return '#6b7280';
  };

  const getEccentricityColor = (value) => {
    if (value <= 2) return '#10b981'; // Green for low
    if (value <= 3) return '#f59e0b'; // Yellow for medium
    return '#ef4444'; // Red for high
  };

  return (
    <div className="metrics-panel">
      <h2>📊 Key Metrics & Rankings</h2>
      
      <div className="metrics-grid">
        {/* Top Hub Accounts */}
        <div className="metric-card hub-card">
          <div className="metric-header">
            <TrendingUp className="metric-icon" />
            <h3>🔵 Top Hub Accounts</h3>
            <p>By Closeness Centrality</p>
          </div>
          <div className="metric-content">
            {topHubAccounts.map((account, index) => (
              <div key={account.Label} className="account-item">
                <div className="rank-badge">#{index + 1}</div>
                <div className="account-info">
                  <span className="account-name">{account.Label}</span>
                  <span className="account-value">
                    {account['Closeness Centrality'].toFixed(3)}
                  </span>
                </div>
                <div 
                  className="metric-bar"
                  style={{
                    backgroundColor: getMetricColor(
                      account['Closeness Centrality'], 
                      0.936524, 
                      'closeness'
                    )
                  }}
                />
              </div>
            ))}
          </div>
        </div>

        {/* Top Bridge Accounts */}
        <div className="metric-card bridge-card">
          <div className="metric-header">
            <Network className="metric-icon" />
            <h3>🟠 Top Bridge Accounts</h3>
            <p>By Betweenness Centrality</p>
          </div>
          <div className="metric-content">
            {topBridgeAccounts.map((account, index) => (
              <div key={account.Label} className="account-item">
                <div className="rank-badge">#{index + 1}</div>
                <div className="account-info">
                  <span className="account-name">{account.Label}</span>
                  <span className="account-value">
                    {account['Betweenness Centrality'].toFixed(3)}
                  </span>
                </div>
                <div 
                  className="metric-bar"
                  style={{
                    backgroundColor: getMetricColor(
                      account['Betweenness Centrality'], 
                      5.946888, 
                      'betweenness'
                    )
                  }}
                />
              </div>
            ))}
          </div>
        </div>

        {/* Peripheral Accounts */}
        <div className="metric-card peripheral-card">
          <div className="metric-header">
            <Target className="metric-icon" />
            <h3>🟡 Peripheral Accounts</h3>
            <p>By Eccentricity</p>
          </div>
          <div className="metric-content">
            {peripheralAccounts.map((account, index) => (
              <div key={account.Label} className="account-item">
                <div className="rank-badge">#{index + 1}</div>
                <div className="account-info">
                  <span className="account-name">{account.Label}</span>
                  <span className="account-value">
                    {account['Eccentricity']}
                  </span>
                </div>
                <div 
                  className="metric-bar"
                  style={{
                    backgroundColor: getEccentricityColor(account['Eccentricity'])
                  }}
                />
              </div>
            ))}
          </div>
        </div>

        {/* Community Overview */}
        <div className="metric-card community-card">
          <div className="metric-header">
            <Users className="metric-icon" />
            <h3>🔗 Community Overview</h3>
            <p>By Modularity Class</p>
          </div>
          <div className="metric-content">
            {Object.entries(communities).map(([community, accounts]) => (
              <div key={community} className="community-item">
                <div className="community-header">
                  <span className="community-name">Community {community}</span>
                  <span className="community-count">{accounts.length} accounts</span>
                </div>
                <div className="community-accounts">
                  {accounts.slice(0, 3).map(account => (
                    <span key={account.Label} className="community-account">
                      {account.Label}
                    </span>
                  ))}
                  {accounts.length > 3 && (
                    <span className="more-accounts">+{accounts.length - 3} more</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MetricsPanel; 