
using System;
using System.Collections.Generic;
using System.Linq;
using DigitalTwin.Models;

namespace DigitalTwin.Services
{
    public class ParkinsonEngine
    {
        private double _baseHr = 70.0;
        private double _baseTemp = 36.6;
        private double _activityLevel = 0.1;
        private int _step = 0;
        public string Scenario { get; private set; } = "RESTING";

        private Position _pos = new() { X = 0, Y = 0, Z = 0 };
        private Position _targetPos = new() { X = 0, Y = 0, Z = 0 };
        
        // Manual control
        public Position ManualMoveVector { get; set; } = new() { X = 0, Y = 0, Z = 0 };
        private bool _isManualMode = false;
        private double _tremorPhase = 0.0;
        private bool _isFreezing = false;
        private int _stepsSinceLastFog = 100; // Track steps since last freeze
        
        private readonly AdvancedAnalyticsEngine _analyticsEngine;
        private readonly EventLoggerService _eventLogger;

        public object[] Obstacles { get; } = new object[]
        {
            new { floor = 0, type = "doorway", x = 0.0, z = -8.0, w = 6.0, d = 6.0, radius = 1.8, name = "Coffee Shop" },
            new { floor = 0, type = "rect", x = 8.0, z = -8.0, w = 6.0, d = 4.0, radius = 2.0, name = "Electronics" },
            new { floor = 0, type = "rect", x = -8.0, z = 8.0, w = 4.0, d = 6.0, radius = 1.8, name = "Clothing" },
            new { floor = 0, type = "circle", x = 0.0, z = 0.0, radius = 2.0, name = "Fountain" },
            new { floor = 1, type = "narrow_hall", x = 0.0, z = -10.0, w = 12.0, d = 4.0, radius = 1.8, name = "Food Court" },
            new { floor = 1, type = "doorway", x = -10.0, z = 0.0, w = 3.0, d = 8.0, radius = 1.5, name = "Restrooms" },
            new { floor = 1, type = "circle", x = 8.0, z = 8.0, radius = 1.5, name = "Lounge Pillar" }
        };

        public object[] Escalators { get; } = new object[]
        {
            new { x = 12.0, z = 0.0, radius = 2.0 }
        };

        public ParkinsonEngine(AdvancedAnalyticsEngine analyticsEngine, EventLoggerService eventLogger)
        {
            _analyticsEngine = analyticsEngine;
            _eventLogger = eventLogger;
            _eventLogger.LogEvent("SYSTEM", "ParkinsonEngine initialized.");
        }

        public void SetScenario(string scenario)
        {
            Scenario = scenario.ToUpper();
            _activityLevel = Scenario switch
            {
                "RUNNING" => 0.8,
                "SLEEPING" => 0.05,
                _ => 0.2
            };
            _eventLogger.LogEvent("SCENARIO", $"Scenario changed to {Scenario}");
        }

        private double GetNoise(double magnitude) => (Random.Shared.NextDouble() - 0.5) * magnitude;

        public TelemetryPacket GenerateTelemetry()
        {
            _step++;
            
            // 1. Parkinsonian Motor Simulation
            double tremorIntensity = UpdateMotorState();
            double proximityToHazard = UpdateSpatialPosition();

            // 2. Vitals Modulation
            double targetHr = Scenario switch
            {
                "RUNNING" => 140.0,
                "SLEEPING" => 55.0,
                "STRESSED" => 95.0,
                _ => 70.0
            };

            // Add stress spikes from FOG and Hazard proximity
            if (_isFreezing) targetHr += 40.0;
            targetHr += proximityToHazard * 35.0;
            targetHr += tremorIntensity * 10.0; // Physical exertion of tremor

            _baseHr += (targetHr - _baseHr) * 0.1;
            _baseTemp += (36.6 + (Scenario == "RUNNING" ? 1.0 : 0) - _baseTemp) * 0.05;

            double finalHr = _baseHr + Math.Sin(_step * 0.2) * 2.0 + GetNoise(3.0);
            
            // Use advanced analytics engine
            var detailedAnalytics = _analyticsEngine.ProcessTelemetry(finalHr, tremorIntensity, _isFreezing, proximityToHazard);

            if (detailedAnalytics.InterventionAuditTrail.Any())
            {
                foreach (var intervention in detailedAnalytics.InterventionAuditTrail)
                {
                    _eventLogger.LogEvent("INTERVENTION", intervention, detailedAnalytics.NeurologicalStressScore);
                }
            }

            var packet = new TelemetryPacket
            {
                Scenario = Scenario,
                Vitals = new VitalsReading
                {
                    HeartRate = Math.Round(finalHr, 1),
                    Temperature = Math.Round(_baseTemp + GetNoise(0.2), 2),
                    ActivityLevel = Math.Round(Math.Max(0, Math.Min(1, _activityLevel + GetNoise(0.05))), 2),
                    Position = new Position { X = Math.Round(_pos.X, 2), Z = Math.Round(_pos.Z, 2) },
                    HazardProximity = Math.Round(proximityToHazard, 2),
                    TremorIntensity = Math.Round(tremorIntensity, 2)
                },
                Analytics = new Analytics
                {
                    StressLevel = detailedAnalytics.NeurologicalStressScore,
                    HealthIndex = Math.Round(100 * (1.0 - detailedAnalytics.NeurologicalStressScore), 1),
                    IsFreezing = detailedAnalytics.FogRiskFlag,
                    Status = detailedAnalytics.NeurologicalStressScore > 0.8 ? "CRITICAL" : detailedAnalytics.NeurologicalStressScore > 0.5 ? "STRESSED" : "EXCELLENT"
                }
            };

            return packet;
        }

        private double UpdateMotorState()
        {
            // tremor logic (5-7Hz)
            _tremorPhase += 0.8; // Oscillation speed
            double tremor = (Math.Sin(_tremorPhase) + GetNoise(0.2)) * 0.5;
            
            // Tremor intensity is higher when stressed or running
            double intensity = (Scenario == "STRESSED" || Scenario == "RUNNING") ? 0.8 : 0.2;
            if (Scenario == "SLEEPING") intensity = 0.0;
            
            return intensity;
        }

        public void SetManualMove(double x, double z)
        {
            ManualMoveVector.X = x;
            ManualMoveVector.Z = z;
            _isManualMode = (Math.Abs(x) > 0.1 || Math.Abs(z) > 0.1);
        }

        private double UpdateSpatialPosition()
        {
            _stepsSinceLastFog++;

            if (_isFreezing)
            {
                if (Random.Shared.Next(0, 50) == 0) 
                {
                    _isFreezing = false;
                    _stepsSinceLastFog = 0; // Reset cooldown after finishing a freeze
                    _eventLogger.LogEvent("MOTOR", "FOG Event Ended", 0.2);
                }
                return 1.0; // High anxiety during FOG
            }

            double speed = Scenario switch
            {
                "RUNNING" => 0.4,
                "SLEEPING" => 0.0,
                _ => 0.15
            };

            // Bradykinesia effect: Parkinson's slowness
            speed *= 0.6; 

            if (_isManualMode)
            {
                // MANUAL CONTROL: Move based on input vector
                _pos.X += ManualMoveVector.X * speed;
                _pos.Z += ManualMoveVector.Z * speed;
            }
            else
            {
                // AUTONOMOUS: Move towards target
                double distToTarget = Math.Sqrt(Math.Pow(_pos.X - _targetPos.X, 2) + Math.Pow(_pos.Z - _targetPos.Z, 2));
                if (distToTarget < 0.5)
                {
                    _targetPos.X = Random.Shared.NextDouble() * 20 - 10;
                    _targetPos.Z = Random.Shared.NextDouble() * 20 - 10;
                }

                _pos.X += (_targetPos.X - _pos.X) * (speed / 2.0); // Slower random walk
                _pos.Z += (_targetPos.Z - _pos.Z) * (speed / 2.0);
            }

            // Collision & FOG logic
            double proximity = 0.0;
            foreach (dynamic obs in Obstacles)
            {
                double obsX = (double)obs.x;
                double obsZ = (double)obs.z;
                double obsRadius = (double)obs.radius;

                double dist = Math.Sqrt(Math.Pow(_pos.X - obsX, 2) + Math.Pow(_pos.Z - obsZ, 2));
                if (dist < obsRadius)
                {
                    proximity = Math.Max(proximity, (obsRadius - dist) / obsRadius);
                    
                    // Simple physics push-out (collision response)
                    double overlap = obsRadius - dist;
                    if (dist > 0.001) // avoid division by zero
                    {
                        _pos.X += ((_pos.X - obsX) / dist) * overlap;
                        _pos.Z += ((_pos.Z - obsZ) / dist) * overlap;
                    }

                    // Possible Freezing of Gait when near obstacles (3% chance per step)
                    // Added Cooldown: Only freeze if it's been at least 100 steps since the last FOG
                    if (dist < obsRadius * 0.8 && _stepsSinceLastFog > 100 && Random.Shared.Next(0, 100) < 3)
                    {
                        _isFreezing = true;
                        _eventLogger.LogEvent("MOTOR", "FOG Event Triggered", 0.9);
                    }
                }
            }

            return proximity;
        }
    }
}
