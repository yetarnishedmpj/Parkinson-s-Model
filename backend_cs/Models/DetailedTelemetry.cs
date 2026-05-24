using System;
using System.Collections.Generic;

namespace DigitalTwin.Models
{
    public class DetailedTelemetry
    {
        public Guid EventId { get; set; } = Guid.NewGuid();
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        public double HrvIndex { get; set; }
        public double MotorFatigueLoad { get; set; }
        public bool FogRiskFlag { get; set; }
        public double NeurologicalStressScore { get; set; }
        public List<string> InterventionAuditTrail { get; set; } = new();
    }
}
