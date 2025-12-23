"""
Core framework sloj - Generička bazna klasa za sve agente
Nema domenskog znanja, samo abstrakcije
"""
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional

# Generički tipovi
TPercept = TypeVar('TPercept')  # Šta agent opaža (npr. slika)
TAction = TypeVar('TAction')  # Šta agent radi (npr. "detekcija")
TResult = TypeVar('TResult')  # Rezultat akcije (npr. "prekršaj pronađen")
TExperience = TypeVar('TExperience')  # Iskustvo za učenje (npr. feedback)


class SoftwareAgent(ABC, Generic[TPercept, TAction, TResult, TExperience]):
    """
    Bazna klasa za agentički ciklus: Sense → Think → Act → Learn

    Svaki agent mora implementirati step_async() metodu koja predstavlja
    jedan "tick" agenta - jednu iteraciju percepcija → odluka → akcija.

    Primjer:
        class ParkingDetectionRunner(SoftwareAgent):
            async def step_async(self):
                # SENSE: Uzmi sliku
                image = await get_image()

                # THINK: Analiziraj
                result = await analyze(image)

                # ACT: Sačuvaj
                await save_result(result)

                return result
    """

    @abstractmethod
    async def step_async(self, cancellation_token=None) -> Optional[TResult]:
        """
        Jedan korak agentičkog ciklusa.

        Returns:
            TResult: Rezultat koraka (ili None ako nema posla)

        Pravila:
        - Korak mora biti atomaran (jedna iteracija)
        - Mora imati "no-work" izlaz bez štete
        - Ne smije sadržavati host logiku (delay, SignalR, CORS)
        - Mora biti idempotentan koliko god može
        """
        pass

    def has_work(self) -> bool:
        """
        Provjerava da li agent ima posla.
        Override ako agent može da provijeri bez izvršavanja step-a.
        """
        return True


class IPerceptionSource(ABC, Generic[TPercept]):
    """
    Izvor percepcija - odakle agent dobija informacije o svijetu
    """

    @abstractmethod
    async def get_next_percept(self) -> Optional[TPercept]:
        """Dohvata sljedeći percept (ili None ako nema)"""
        pass


class IPolicy(ABC, Generic[TPercept, TAction]):
    """
    Politika odlučivanja - kako agent bira akciju
    """

    @abstractmethod
    def decide(self, percept: TPercept) -> TAction:
        """Na osnovu percepta odlučuje koju akciju izvršiti"""
        pass


class IActuator(ABC, Generic[TAction, TResult]):
    """
    Aktuator - izvršava akcije i vraća rezultat
    """

    @abstractmethod
    async def execute(self, action: TAction) -> TResult:
        """Izvršava akciju i vraća rezultat"""
        pass


class ILearningComponent(ABC, Generic[TExperience]):
    """
    Komponenta za učenje - ažurira znanje agenta
    """

    @abstractmethod
    async def learn(self, experience: TExperience) -> None:
        """Uči iz iskustva (npr. ažurira model, brojače, metriku)"""
        pass