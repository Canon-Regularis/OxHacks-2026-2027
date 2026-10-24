from datetime import UTC, datetime, timedelta

import pytest
from gridpulse.analysis.weather import weather_statistics

START = datetime(2026, 10, 23, tzinfo=UTC)


def rows(count=200):
    begin = START - timedelta(hours=count)
    # Nonconstant differences with a known exact linear relationship.
    weather = []
    prices = []
    for i in range(count + 25):
        at = begin + timedelta(hours=i)
        value = float(i * i)
        weather.append({'bucket': at, 'temperature': value, 'pressure': 1000 + value / 100})
        prices.append({'bucket': at, 'price': value * 2})
    return weather, prices


def test_seven_day_changes_correlate_and_do_not_read_the_future():
    weather, prices = rows()
    result = weather_statistics(weather, prices, START, START + timedelta(days=1))
    first = result[0]
    assert first.t == START
    assert first.temperature_samples == 168
    assert first.temperature_correlation == pytest.approx(1)
    # t=midnight uses the preceding complete hour, not the midnight-start observation.
    assert first.temperature == 199**2
    prices[-1]['price'] = -1e9
    again = weather_statistics(weather, prices, START, START + timedelta(days=1))
    assert again[0] == first


def test_missing_hour_is_not_a_multi_hour_change():
    weather, prices = rows()
    weather = [row for row in weather if row['bucket'] != START - timedelta(hours=2)]
    result = weather_statistics(weather, prices, START, START + timedelta(hours=1))[0]
    assert result.temperature_change is None
    assert result.temperature_samples == 166


def test_constant_or_insufficient_data_returns_null_correlation():
    weather, prices = rows(5)
    for row in weather:
        row['temperature'] = 5
    result = weather_statistics(weather, prices, START, START + timedelta(hours=1))[0]
    assert result.temperature_correlation is None
    assert result.pressure_correlation is None


def test_empty_data_preserves_hour_grid_and_missing_values():
    result = weather_statistics([], [], START, START + timedelta(days=1))
    assert len(result) == 24
    assert all(point.temperature is None and point.price_change is None for point in result)
    assert all(point.temperature_samples == 0 for point in result)
