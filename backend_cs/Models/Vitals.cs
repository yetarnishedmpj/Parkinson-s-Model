
using System;

namespace DigitalTwin.Models
{
    public class Position
    {
        public double X { get; set; }
        public double Y { get; set; }
        public double Z { get; set; }
        public int Floor { get; set; } = 0;
    }

    public class VitalsReading
    {
        public double HeartRate { get; set; }
        public double Temperature { get; set; }
        public double ActivityLevel { get; set; }
        public Position Position { get; set; } = new();
        public double HazardProximity { get; set; }
        public double TremorIntensity { get; set; }
        public string Timestamp { get; set; } = DateTime.UtcNow.ToString("o");
    }

    public class Analytics
    {
        public double StressLevel { get; set; }
        public double HealthIndex { get; set; }
        public string Status { get; set; } = "Unknown";
        public bool IsFreezing { get; set; }
    }

    public class TelemetryPacket
    {
        public VitalsReading Vitals { get; set; } = new();
        public Analytics Analytics { get; set; } = new();
        public string Scenario { get; set; } = "RESTING";
    }
}
