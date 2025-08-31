import React from 'react';
import MetricsPanel from './MetricsPanel';
import AlertsSection from './AlertsSection';
import './Dashboard.css';

const Dashboard = ({ data }) => {
  // Calculate top accounts by different metrics
  const topHubAccounts = [...data]
    .sort((a, b) => b['Closeness Centrality'] - a['Closeness Centrality'])
    .slice(0, 3);

  const topBridgeAccounts = [...data]
    .sort((a, b) => b['Betweenness Centrality'] - a['Betweenness Centrality'])
    .slice(0, 3);

  const peripheralAccounts = [...data]
    .sort((a, b) => b['Eccentricity'] - a['Eccentricity'])
    .slice(0, 5);

  // Group accounts by modularity class
  const communities = data.reduce((acc, account) => {
    const community = account['Modularity Class'];
    if (!acc[community]) acc[community] = [];
    acc[community].push(account);
    return acc;
  }, {});

  // Calculate alerts
  const alerts = [];
  
  // Hub alerts (high closeness)
  topHubAccounts.forEach(account => {
    if (account['Closeness Centrality'] > 0.8) {
      alerts.push({
        type: 'hub',
        message: `🔵 Hub Alert: ${account.Label} is highly connected across platforms`,
        severity: 'high',
        data: account
      });
    }
  });

  // Bridge alerts (high betweenness)
  topBridgeAccounts.forEach(account => {
    if (account['Betweenness Centrality'] > 3.0) {
      alerts.push({
        type: 'bridge',
        message: `🟠 Bridge Alert: ${account.Label} links different alias clusters`,
        severity: 'medium',
        data: account
      });
    }
  });

  // Fringe alerts (high eccentricity)
  peripheralAccounts.forEach(account => {
    if (account['Eccentricity'] > 2) {
      alerts.push({
        type: 'fringe',
        message: `🟡 Fringe Alert: ${account.Label} appears isolated`,
        severity: 'low',
        data: account
      });
    }
  });

  // Subgroup alerts (small communities)
  Object.entries(communities).forEach(([community, accounts]) => {
    if (accounts.length < 5) {
      alerts.push({
        type: 'subgroup',
        message: `🟢 Subgroup Alert: Community ${community} has only ${accounts.length} accounts`,
        severity: 'info',
        data: { community, count: accounts.length, accounts }
      });
    }
  });

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>🛡️ Cyber Shield OSINT Dashboard</h1>
        <p>Identity Intelligence & Network Analysis</p>
        <div className="stats-summary">
          <span>📊 {data.length} Accounts Analyzed</span>
          <span>🔗 {Object.keys(communities).length} Communities Detected</span>
          <span>🚨 {alerts.length} Alerts Generated</span>
        </div>
      </header>

      <div className="dashboard-content">
        <MetricsPanel 
          topHubAccounts={topHubAccounts}
          topBridgeAccounts={topBridgeAccounts}
          peripheralAccounts={peripheralAccounts}
          communities={communities}
        />
        
        <div className="dashboard-grid">
          <AlertsSection alerts={alerts} />
        </div>
      </div>
    </div>
  );
};

export default Dashboard; 