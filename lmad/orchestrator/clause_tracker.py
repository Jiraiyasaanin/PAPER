from typing import List, Optional, Set
from lmad.models.schemas import Clause


class ClauseTracker:
    """Manages strictly partitioned pools of For and Against evidence clauses,
    tracking which clauses have been introduced/debated.
    """

    def __init__(self, for_clauses: List[Clause], against_clauses: List[Clause]):
        self._all_for: dict[str, Clause] = {c.id: c for c in for_clauses}
        self._all_against: dict[str, Clause] = {c.id: c for c in against_clauses}

        self._used_for_ids: Set[str] = set()
        self._used_against_ids: Set[str] = set()

    def get_unused_for(self) -> List[Clause]:
        """Returns the list of remaining unused clauses in the 'For' pool."""
        return [c for cid, c in self._all_for.items() if cid not in self._used_for_ids]

    def get_unused_against(self) -> List[Clause]:
        """Returns the list of remaining unused clauses in the 'Against' pool."""
        return [c for cid, c in self._all_against.items() if cid not in self._used_against_ids]

    def mark_for_used(self, clause_ids: List[str]) -> None:
        """Marks specific For clause IDs as used."""
        for cid in clause_ids:
            if cid in self._all_for:
                self._used_for_ids.add(cid)

    def mark_against_used(self, clause_ids: List[str]) -> None:
        """Marks specific Against clause IDs as used."""
        for cid in clause_ids:
            if cid in self._all_against:
                self._used_against_ids.add(cid)

    def has_remaining_evidence(self) -> bool:
        """Returns True if there is at least one unused clause in either pool."""
        return len(self.get_unused_for()) > 0 or len(self.get_unused_against()) > 0

    def all_evidence_presented(self) -> bool:
        """Returns True when both the 'For' and 'Against' agents have presented all their evidence clauses."""
        return len(self.get_unused_for()) == 0 and len(self.get_unused_against()) == 0

    def get_for_clause(self, clause_id: str) -> Optional[Clause]:
        return self._all_for.get(clause_id)

    def get_against_clause(self, clause_id: str) -> Optional[Clause]:
        return self._all_against.get(clause_id)

    @property
    def all_for_clauses(self) -> List[Clause]:
        return list(self._all_for.values())

    @property
    def all_against_clauses(self) -> List[Clause]:
        return list(self._all_against.values())
