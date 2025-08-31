import React, { useState } from 'react';
import { AlertTriangle, Info, ChevronDown, ChevronUp } from 'lucide-react';
import './AlertsSection.css';

const AlertsSection = ({ alerts }) => {
  const [expandedAlerts, setExpandedAlerts] = useState(new Set());

  const toggleAlert = (index) => {
    const newExpanded = new Set(expandedAlerts);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedAlerts(newExpanded);
  };

  const getAlertIcon = (type) => {
    switch (type) {
      case 'hub':
        return '🔵';
      case 'bridge':
        return '🟠';
      case 'fringe':
        return '🟡';
      case 'subgroup':
        return '🟢';
      default:
        return 'ℹ️';
    }
  };

  const getAlertColor = (type) => {
    switch (type) {
      case 'hub':
        return '#3b82f6';
      case 'bridge':
        return '#f97316';
      case 'fringe':
        return '#eab308';
      case 'subgroup':
        return '#10b981';
      default:
        return '#6b7280';
    }
  };

  const getSeverityIcon = (severity) => {
    switch (severity) {
      case 'high':
        return <AlertTriangle className="severity-icon high" />;
      case 'medium':
        return <AlertTriangle className="severity-icon medium" />;
      case 'low':
        return <Info className="severity-icon low" />;
      case 'info':
        return <Info className="severity-icon info" />;
      default:
        return <Info className="severity-icon info" />;
    }
  };

  const renderAlertDetails = (alert) => {
    switch (alert.type) {
      case 'hub':
        return (
          <div className="alert-details">
            <div className="detail-row">
              <span className="detail-label">Closeness Centrality:</span>
              <span className="detail-value">{alert.data['Closeness Centrality'].toFixed(3)}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Betweenness Centrality:</span>
              <span className="detail-value">{alert.data['Betweenness Centrality'].toFixed(3)}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Eccentricity:</span>
              <span className="detail-value">{alert.data['Eccentricity']}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Community:</span>
              <span className="detail-value">{alert.data['Modularity Class']}</span>
            </div>
          </div>
        );
      
      case 'bridge':
        return (
          <div className="alert-details">
            <div className="detail-row">
              <span className="detail-label">Betweenness Centrality:</span>
              <span className="detail-value">{alert.data['Betweenness Centrality'].toFixed(3)}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Closeness Centrality:</span>
              <span className="detail-value">{alert.data['Closeness Centrality'].toFixed(3)}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Role:</span>
              <span className="detail-value">Critical connector between communities</span>
            </div>
          </div>
        );
      
      case 'fringe':
        return (
          <div className="alert-details">
            <div className="detail-row">
              <span className="detail-label">Eccentricity:</span>
              <span className="detail-value">{alert.data['Eccentricity']}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Closeness Centrality:</span>
              <span className="detail-value">{alert.data['Closeness Centrality'].toFixed(3)}</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Risk Level:</span>
              <span className="detail-value">Isolated account - monitor for activity</span>
            </div>
          </div>
        );
      
      case 'subgroup':
        return (
          <div className="alert-details">
            <div className="detail-row">
              <span className="detail-label">Community Size:</span>
              <span className="detail-value">{alert.data.count} accounts</span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Members:</span>
              <span className="detail-value">
                {alert.data.accounts.map(acc => acc.Label).join(', ')}
              </span>
            </div>
            <div className="detail-row">
              <span className="detail-label">Analysis:</span>
              <span className="detail-value">Small alias family - potential investigation target</span>
            </div>
          </div>
        );
      
      default:
        return null;
    }
  };

  if (alerts.length === 0) {
    return (
      <div className="alerts-section">
        <h2>🚨 Alerts & Intelligence</h2>
        <div className="no-alerts">
          <Info className="no-alerts-icon" />
          <p>No alerts generated. All accounts appear to be within normal parameters.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="alerts-section">
      <h2>🚨 Alerts & Intelligence</h2>
      <div className="alerts-container">
        {alerts.map((alert, index) => (
          <div 
            key={index} 
            className={`alert-item ${alert.type}`}
            style={{ borderLeftColor: getAlertColor(alert.type) }}
          >
            <div className="alert-header">
              <div className="alert-icon">
                {getAlertIcon(alert.type)}
              </div>
              <div className="alert-content">
                <div className="alert-message">{alert.message}</div>
                <div className="alert-meta">
                  {getSeverityIcon(alert.severity)}
                  <span className="severity-text">{alert.severity.toUpperCase()}</span>
                  <span className="alert-type">{alert.type.toUpperCase()}</span>
                </div>
              </div>
              <button 
                className="expand-button"
                onClick={() => toggleAlert(index)}
              >
                {expandedAlerts.has(index) ? <ChevronUp /> : <ChevronDown />}
              </button>
            </div>
            
            {expandedAlerts.has(index) && (
              <div className="alert-body">
                {renderAlertDetails(alert)}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default AlertsSection; 