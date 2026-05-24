using System;
using System.Collections.Generic;
using System.Linq;
using DigitalTwin.Models;

namespace DigitalTwin.Services
{
    public class AdvancedAnalyticsEngine
    {
        private readonly Queue<double> _hrHistory = new();
        private readonly Queue<double> _tremorHistory = new();
        private const int WindowSize = 50;

        public DetailedTelemetry ProcessTelemetry(double hr, double tremor, bool isFreezing, double proximity)
        {
            _hrHistory.Enqueue(hr);
            if (_hrHistory.Count > WindowSize) _hrHistory.Dequeue();

            _tremorHistory.Enqueue(tremor);
            if (_tremorHistory.Count > WindowSize) _tremorHistory.Dequeue();

            // Calculate HRV (Root Mean Square of Successive Differences approximation)
            double hrv = 0;
            if (_hrHistory.Count > 1)
            {
                var hrArray = _hrHistory.ToArray();
                double sumSqDiff = 0;
                for (int i = 1; i < hrArray.Length; i++)
                {
                    double diff = hrArray[i] - hrArray[i - 1];
                    sumSqDiff += diff * diff;
                }
                hrv = Math.Sqrt(sumSqDiff / (hrArray.Length - 1));
            }

            // Motor fatigue load based on sustained tremor
            double avgTremor = _tremorHistory.Average();
            double motorFatigue = Math.Clamp(avgTremor * 1.5 + (isFreezing ? 0.3 : 0), 0, 1.0);

            // Neurological Stress Score (Complex heuristic)
            double neuroStress = Math.Clamp((hr - 60) / 100.0 * 0.4 + (1.0 / (hrv + 1)) * 0.3 + proximity * 0.3, 0, 1.0);
            if (isFreezing) neuroStress += 0.5;

            var telemetry = new DetailedTelemetry
            {
                HrvIndex = Math.Round(hrv, 2),
                MotorFatigueLoad = Math.Round(motorFatigue, 3),
                FogRiskFlag = isFreezing || avgTremor > 0.75,
                NeurologicalStressScore = Math.Round(neuroStress, 3)
            };

            if (telemetry.FogRiskFlag)
                telemetry.InterventionAuditTrail.Add("FOG_RISK_ELEVATED");
            
            if (hrv < 2.0 && _hrHistory.Count == WindowSize)
                telemetry.InterventionAuditTrail.Add("HRV_DEPRESSED_WARNING");

            return telemetry;
        }
    }
}
