from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter

def setup_telemetry():
    """
    Initialize OpenTelemetry for the Noisefloor/RiskLayer platform.
    This routes traces to the console for demonstration, but in production 
    would route to Datadog, New Relic, or Jaeger via OTLP.
    """
    provider = TracerProvider()
    
    # In a real enterprise setup, use BatchSpanProcessor with OTLPSpanExporter
    processor = SimpleSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)
    
    trace.set_tracer_provider(provider)

def get_tracer(name: str = "noisefloor"):
    return trace.get_tracer(name)
