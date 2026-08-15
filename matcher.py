import math
from dataclasses import dataclass

import cv2
import numpy as np

from math import dist


@dataclass(frozen=True)
class TemplateMatchResult:
    matches: list[list[int]]
    best_confidence: float
    raw_match_count: int = 0
    best_match: list[int] | None = None
    template_scale: float = 1.0


class Matcher:

    def __init__(self, group_threshold=1, eps=0.2):
        self.group_threshold = group_threshold
        self.eps = eps

    def match_template(self, template, target, matching_threshold=0.45, grouping=True):
        return self.match_template_details(
            template,
            target,
            matching_threshold,
            grouping,
        ).matches

    def match_template_details(
        self,
        template,
        target,
        matching_threshold=0.45,
        grouping=True,
    ):
        result = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
        _, best_confidence, _, best_location = cv2.minMaxLoc(result)
        w = template.shape[1]
        h = template.shape[0]
        yloc, xloc = np.where(result >= matching_threshold)

        matches = []
        confidences = []
        for (x, y) in zip(xloc, yloc):
            matches.append([int(x + w / 2), int(y + h / 2), int(w), int(h)])
            confidences.append(float(result[y, x]))

        if grouping and matches:
            matches = self._suppress_overlaps(matches, confidences)
        normalized_matches = [list(map(int, match)) for match in matches]
        best_match = [
            int(best_location[0] + w / 2),
            int(best_location[1] + h / 2),
            int(w),
            int(h),
        ]
        return TemplateMatchResult(
            normalized_matches,
            float(best_confidence),
            len(xloc),
            best_match,
        )

    def match_template_multiscale(
        self,
        template,
        target,
        matching_threshold=0.45,
        scales=(1.0,),
        grouping=True,
    ):
        best_result = None
        for scale in dict.fromkeys(scales):
            if scale <= 0:
                raise ValueError("template scales must be greater than zero")
            width = max(1, round(template.shape[1] * scale))
            height = max(1, round(template.shape[0] * scale))
            if width > target.shape[1] or height > target.shape[0]:
                continue
            interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            scaled_template = cv2.resize(
                template,
                (width, height),
                interpolation=interpolation,
            )
            result = self.match_template_details(
                scaled_template,
                target,
                matching_threshold,
                grouping,
            )
            scaled_result = TemplateMatchResult(
                result.matches,
                result.best_confidence,
                result.raw_match_count,
                result.best_match,
                float(scale),
            )
            if best_result is None or scaled_result.best_confidence > best_result.best_confidence:
                best_result = scaled_result

        if best_result is None:
            raise ValueError("no template scale fits inside the target image")
        return best_result

    def _suppress_overlaps(self, matches, confidences):
        ranked = sorted(
            zip(matches, confidences),
            key=lambda item: item[1],
            reverse=True,
        )
        selected = []
        for candidate, _ in ranked:
            if all(
                self._intersection_over_union(candidate, existing) <= self.eps
                for existing in selected
            ):
                selected.append(candidate)
        return selected

    @staticmethod
    def _intersection_over_union(first, second):
        first_left = first[0] - first[2] / 2
        first_top = first[1] - first[3] / 2
        first_right = first[0] + first[2] / 2
        first_bottom = first[1] + first[3] / 2
        second_left = second[0] - second[2] / 2
        second_top = second[1] - second[3] / 2
        second_right = second[0] + second[2] / 2
        second_bottom = second[1] + second[3] / 2

        intersection_width = max(0, min(first_right, second_right) - max(first_left, second_left))
        intersection_height = max(0, min(first_bottom, second_bottom) - max(first_top, second_top))
        intersection = intersection_width * intersection_height
        if intersection == 0:
            return 0.0
        first_area = first[2] * first[3]
        second_area = second[2] * second[3]
        return intersection / (first_area + second_area - intersection)

    def match_template_exists(self, template, target, matching_threshold=0.45):
        result = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
        if len(result) == 0:
            return False
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        return max_val > matching_threshold

    def matchs_to_boundary(self, matches, tolerance=50):
        left = min(matches, key=lambda m: m[0])
        right = max(matches, key=lambda m: m[0])
        top = min(matches, key=lambda m: m[1])
        bottom = max(matches, key=lambda m: m[1])
        return (
            (top[0], top[1] - tolerance),
            (left[0] - tolerance * 2, left[1]),
            (bottom[0], bottom[1] + tolerance),
            (right[0] + tolerance * 2, right[1]))

    def boundary_to_path(self, boundary, thickness=55):
        top, left, bottom, right = boundary
        path = [top, left]
        for i in range(1, math.ceil(dist(top, right) / thickness)):
            ta = np.sqrt(thickness**2 / 5)
            path.append((int(top[0] + 2*ta*i), int(top[1] + ta*i)))
            path.append((int(left[0] + 2*ta*i), int(left[1] + ta*i)))
        return path

    def mark_matches(self, matches, target, color):
        for (x, y, w, h) in matches:
            cv2.circle(target, (x, y), 2, color, 2)
            cv2.rectangle(target, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), color, 2)

    def mark_boundary(self, boundary, target):
        # TODO: refactor target, extract to constructor
        top, left, bottom, right = boundary
        cv2.line(target, top, left, (0, 0, 0), 2)
        cv2.line(target, left, bottom, (0, 0, 0), 2)
        cv2.line(target, bottom, right, (0, 0, 0), 2)
        cv2.line(target, right, top, (0, 0, 0), 2)

    def mark_path(self, points, target):
        before = -1
        for p in points:
            if before != -1:
                cv2.line(target, before, p, (0, 0, 0), 2)
            cv2.circle(target, p, 2, (0, 0, 255), 2)
            before = p
