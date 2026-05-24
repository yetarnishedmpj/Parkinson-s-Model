
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.FileProviders;
using System.Net.WebSockets;
using System.Text.Json;
using System.Text;
using DigitalTwin.Services;
using DigitalTwin.Models;

var builder = WebApplication.CreateBuilder(args);

// Add services
builder.Services.AddSingleton<EventLoggerService>();
builder.Services.AddSingleton<AdvancedAnalyticsEngine>();
builder.Services.AddSingleton<ParkinsonEngine>();
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowAll", p => p.AllowAnyOrigin().AllowAnyMethod().AllowAnyHeader());
});

var app = builder.Build();

app.UseCors("AllowAll");
app.UseWebSockets();

// Enable Static File Serving for Frontend
var frontendPath = Path.GetFullPath(Path.Combine(app.Environment.ContentRootPath, "..", "frontend"));
app.UseFileServer(new FileServerOptions
{
    FileProvider = new PhysicalFileProvider(frontendPath),
    RequestPath = "",
    EnableDefaultFiles = true
});

// Scenario Endpoint
app.MapPost("/api/scenario/{name}", (string name, ParkinsonEngine engine) =>
{
    engine.SetScenario(name);
    return Results.Ok(new { status = "success", scenario = engine.Scenario });
});

// Manual Control Endpoint
app.MapPost("/api/control/move", (Position move, ParkinsonEngine engine) =>
{
    engine.SetManualMove(move.X, move.Z);
    return Results.Ok(new { status = "success" });
});

// WebSocket Telemetry Endpoint
app.Map("/ws", async context =>
{
    if (context.WebSockets.IsWebSocketRequest)
    {
        using var webSocket = await context.WebSockets.AcceptWebSocketAsync();
        var engine = app.Services.GetRequiredService<ParkinsonEngine>();
        
        while (webSocket.State == WebSocketState.Open)
        {
            var telemetry = engine.GenerateTelemetry();
            var json = JsonSerializer.Serialize(telemetry, new JsonSerializerOptions 
            { 
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase 
            });
            var bytes = Encoding.UTF8.GetBytes(json);
            
            await webSocket.SendAsync(
                new ArraySegment<byte>(bytes), 
                WebSocketMessageType.Text, 
                true, 
                CancellationToken.None);
            
            await Task.Delay(100); // 10Hz for smooth motion
        }
    }
    else
    {
        context.Response.StatusCode = StatusCodes.Status400BadRequest;
    }
});

app.Run();
