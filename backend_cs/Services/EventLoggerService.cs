using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;

namespace DigitalTwin.Services
{
    public class EventLoggerService
    {
        private readonly ConcurrentQueue<LogEntry> _eventLog = new();

        public void LogEvent(string category, string message, double severity = 0.0)
        {
            var entry = new LogEntry
            {
                Id = Guid.NewGuid(),
                Timestamp = DateTime.UtcNow,
                Category = category,
                Message = message,
                Severity = severity
            };
            
            _eventLog.Enqueue(entry);
            
            if (_eventLog.Count > 1000)
            {
                _eventLog.TryDequeue(out _);
            }
            
            Console.WriteLine($"[AUDIT] {entry.Timestamp:O} | {category} | {message} | Sev: {severity:F2}");
        }

        public IEnumerable<LogEntry> GetRecentEvents(int count = 50)
        {
            return _eventLog.Reverse().Take(count);
        }
    }

    public class LogEntry
    {
        public Guid Id { get; set; }
        public DateTime Timestamp { get; set; }
        public string Category { get; set; }
        public string Message { get; set; }
        public double Severity { get; set; }
    }
}
