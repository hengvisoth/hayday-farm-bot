"""Pure coordinate planning for adaptive harvesting."""

from dataclasses import dataclass
from math import ceil, dist
from statistics import median
from typing import Mapping, Sequence

Point = tuple[int, int]
Match = Sequence[int]


@dataclass(frozen=True)
class CameraTranslation:
    dx: int
    dy: int
    method: str
    confidence: str


@dataclass(frozen=True)
class FieldSweepPlan:
    bounds: tuple[int, int, int, int]
    rows: int
    route: list[Point]


def _distance_squared(first: Point, second: Point) -> int:
    return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2


def select_center_match(matches: Sequence[Match]):
    if not matches:
        return None
    mean_x = sum(match[0] for match in matches) / len(matches)
    mean_y = sum(match[1] for match in matches) / len(matches)
    return min(
        matches,
        key=lambda match: (match[0] - mean_x) ** 2 + (match[1] - mean_y) ** 2,
    )


def select_nearest_match(matches: Sequence[Match], point: Point, max_distance: float | None = None):
    if not matches:
        return None
    nearest = min(matches, key=lambda match: _distance_squared((match[0], match[1]), point))
    if max_distance is not None:
        if _distance_squared((nearest[0], nearest[1]), point) > max_distance**2:
            return None
    return nearest


def _mutual_nearest_pairs(
    before: Sequence[Point],
    after: Sequence[Point],
    maximum_distance: float,
) -> list[tuple[Point, Point]]:
    if not before or not after:
        return []

    pairs = []
    limit_squared = maximum_distance**2
    for before_point in before:
        after_point = min(after, key=lambda point: _distance_squared(before_point, point))
        if _distance_squared(before_point, after_point) > limit_squared:
            continue
        reverse_point = min(before, key=lambda point: _distance_squared(after_point, point))
        if reverse_point == before_point:
            pairs.append((before_point, after_point))
    return pairs


def _validated_translation(
    dx: int,
    dy: int,
    method: str,
    confidence: str,
    screen_size: Point,
) -> CameraTranslation:
    width, height = screen_size
    if abs(dx) > width * 0.25 or abs(dy) > height * 0.25:
        return CameraTranslation(0, 0, "stable_fallback", f"rejected_{method}")
    return CameraTranslation(dx, dy, method, confidence)


def estimate_camera_translation(
    before_anchors: Mapping[str, Point],
    after_anchors: Mapping[str, Point],
    before_crops: Sequence[Point],
    after_crops: Sequence[Point],
    screen_size: Point,
) -> CameraTranslation:
    for name in ("boat", "market"):
        if name not in before_anchors or name not in after_anchors:
            continue
        before = before_anchors[name]
        after = after_anchors[name]
        return _validated_translation(
            after[0] - before[0],
            after[1] - before[1],
            "shared_anchor",
            name,
            screen_size,
        )

    pairs = _mutual_nearest_pairs(before_crops, after_crops, maximum_distance=200)
    if pairs:
        deltas = [
            (after[0] - before[0], after[1] - before[1])
            for before, after in pairs
        ]
        median_dx = round(median(delta[0] for delta in deltas))
        median_dy = round(median(delta[1] for delta in deltas))
        reliable = [
            delta
            for delta in deltas
            if abs(delta[0] - median_dx) <= 15
            and abs(delta[1] - median_dy) <= 15
        ]
        if reliable:
            reliable_dx = round(median(delta[0] for delta in reliable))
            reliable_dy = round(median(delta[1] for delta in reliable))
            return _validated_translation(
                reliable_dx,
                reliable_dy,
                "crop_positions",
                f"{len(reliable)}/{len(pairs)}_pairs",
                screen_size,
            )

    return CameraTranslation(0, 0, "stable_fallback", "no_reliable_movement_evidence")


def translate_points(
    points: Sequence[Point], translation: CameraTranslation
) -> list[Point]:
    return [(x + translation.dx, y + translation.dy) for x, y in points]


def _interpolate_segment(start: Point, end: Point, max_segment: float) -> list[Point]:
    steps = max(1, ceil(dist(start, end) / max_segment))
    return [
        (
            round(start[0] + (end[0] - start[0]) * step / steps),
            round(start[1] + (end[1] - start[1]) * step / steps),
        )
        for step in range(1, steps + 1)
    ]


def build_drag_route(
    crop_centers: Sequence[Point],
    start: Point,
    max_segment: float = 25,
) -> list[Point]:
    if max_segment <= 0:
        raise ValueError("max_segment must be greater than zero")

    remaining = list(dict.fromkeys(crop_centers))
    route = []
    current = start
    while remaining:
        next_crop = min(remaining, key=lambda point: _distance_squared(current, point))
        route.extend(_interpolate_segment(current, next_crop, max_segment))
        current = next_crop
        remaining.remove(next_crop)
    return route


def build_field_route(
    matches: Sequence[Match],
    start: Point,
    row_start: Point | None = None,
    max_segment: float = 25,
) -> list[Point]:
    """Visit every matched tile via a row-by-row sweep, not nearest-neighbor.

    Nearest-neighbor routing can zigzag back and forth across a field,
    producing short back-and-forth segments that touch-input recognition
    can filter out as jitter instead of a deliberate drag. Grouping the
    real matched points into rows and sweeping each row in one direction
    (alternating like mowing a lawn) keeps the motion monotonic while
    still landing on every detected tile exactly.

    ``start`` is where the drag physically begins (e.g. the tool icon
    being held down), so the route always starts there. ``row_start``
    (defaults to ``start``) only decides which row/corner is swept
    first -- pass the field tile the user actually selected so the fill
    begins there instead of wherever happens to be nearest the tool icon.
    """
    if not matches:
        return []

    points = list(dict.fromkeys((match[0], match[1]) for match in matches))
    typical_height = max(1, round(median(match[3] for match in matches)))
    row_gap = max(1, round(typical_height / 2))

    by_y = sorted(points, key=lambda point: point[1])
    rows = [[by_y[0]]]
    for point in by_y[1:]:
        if point[1] - rows[-1][-1][1] <= row_gap:
            rows[-1].append(point)
        else:
            rows.append([point])

    for row in rows:
        row.sort(key=lambda point: point[0])

    anchor = start if row_start is None else row_start
    # Pick a sweep direction by finding the row actually nearest the
    # anchor, not just comparing against the field's two extreme rows --
    # a multi-row field selected near a middle row would otherwise pick
    # whichever extreme happens to be marginally closer, ignoring that
    # the real selection was elsewhere entirely.
    nearest_row_index = min(
        range(len(rows)),
        key=lambda index: min(_distance_squared(anchor, point) for point in rows[index]),
    )
    if nearest_row_index > (len(rows) - 1) / 2:
        rows.reverse()

    first_row = rows[0]
    left_to_right = _distance_squared(anchor, first_row[0]) <= _distance_squared(
        anchor,
        first_row[-1],
    )

    ordered_points = []
    for row_index, row in enumerate(rows):
        move_left_to_right = left_to_right if row_index % 2 == 0 else not left_to_right
        ordered_points.extend(row if move_left_to_right else list(reversed(row)))

    route = []
    current = start
    for point in ordered_points:
        route.extend(_interpolate_segment(current, point, max_segment))
        current = point
    return route


def _bounds(matches: Sequence[Match]) -> tuple[int, int, int, int]:
    left = round(min(match[0] - match[2] / 2 for match in matches))
    top = round(min(match[1] - match[3] / 2 for match in matches))
    right = round(max(match[0] + match[2] / 2 for match in matches))
    bottom = round(max(match[1] + match[3] / 2 for match in matches))
    return left, top, right, bottom


def select_field_center(matches: Sequence[Match]):
    """Return the midpoint of the matches' bounding box, not a matched tile.

    A sparse or edge-heavy set of matches (e.g. only the field's corners)
    can leave every individual match sitting outside the field's actual
    clickable interior. The bounding-box midpoint stays inside any convex
    field shape regardless of which tiles were matched.
    """
    if not matches:
        return None
    left, top, right, bottom = _bounds(matches)
    return (round((left + right) / 2), round((top + bottom) / 2))


def build_field_sweep_plan(
    field_matches: Sequence[Match],
    start: Point,
) -> FieldSweepPlan:
    if not field_matches:
        return FieldSweepPlan((0, 0, 0, 0), 0, [])

    left, top, right, bottom = _bounds(field_matches)

    typical_height = max(1, round(median(match[3] for match in field_matches)))
    maximum_row_gap = max(1, round(typical_height / 2))
    row_intervals = max(1, ceil((bottom - top) / maximum_row_gap))
    row_positions = [
        round(top + (bottom - top) * index / row_intervals)
        for index in range(row_intervals + 1)
    ]

    top_distance = min(
        _distance_squared(start, (left, top)),
        _distance_squared(start, (right, top)),
    )
    bottom_distance = min(
        _distance_squared(start, (left, bottom)),
        _distance_squared(start, (right, bottom)),
    )
    if bottom_distance < top_distance:
        row_positions.reverse()

    first_y = row_positions[0]
    left_to_right = _distance_squared(start, (left, first_y)) <= _distance_squared(
        start,
        (right, first_y),
    )
    route = []
    for row_index, y in enumerate(row_positions):
        move_left_to_right = left_to_right if row_index % 2 == 0 else not left_to_right
        if move_left_to_right:
            route.extend([(left, y), (right, y)])
        else:
            route.extend([(right, y), (left, y)])

    return FieldSweepPlan((left, top, right, bottom), len(row_positions), route)
