# SPDX-License-Identifier: AGPL-3.0-or-later
"""Processors for engine-type: ``offline``"""

import typing as t

from .abstract import EngineProcessor, RequestParams

if t.TYPE_CHECKING:
    from searx.results import ResultContainer


class OfflineProcessor(EngineProcessor):
    """Processor class used by ``offline`` engines."""

    engine_type: str = "offline"

    def search(
        self,
        query: str,
        params: RequestParams,
        result_container: "ResultContainer",
        start_time: float,
        timeout_limit: float,
    ) -> None:
        try:
            search_results = self.engine.search(query, params)
            self.extend_container(result_container, start_time, search_results)
        except ValueError as e:
            # do not record the error
            self.logger.exception(
                f"engine {self.engine.name} : invalid input : {e}"
            )
        except Exception as e:  # pylint: disable=broad-except
            self.handle_exception(result_container, e)
            self.logger.exception(
                f"engine {self.engine.name} : exception : {e}"
            )
