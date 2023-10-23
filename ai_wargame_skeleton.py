from __future__ import annotations
import argparse
import copy
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from io import TextIOWrapper
from time import sleep
from typing import Tuple, TypeVar, Type, Iterable, ClassVar
import random
import requests

from shutil import copyfile

# maximum and minimum values for our heuristic scores (usually represents an end of game condition)
MAX_HEURISTIC_SCORE = 2000000000
MIN_HEURISTIC_SCORE = -2000000000

#create the file and allow to append to the file while the game is running (close it only at the end of main)
gf = open("output.txt", "w")

class UnitType(Enum):
    """Every unit type."""
    AI = 0
    Tech = 1
    Virus = 2
    Program = 3
    Firewall = 4

class Player(Enum):
    """The 2 players."""
    Attacker = 0
    Defender = 1

    def next(self) -> Player:
        """The next (other) player."""
        if self is Player.Attacker:
            return Player.Defender
        else:
            return Player.Attacker

class GameType(Enum):
    AttackerVsDefender = 0
    AttackerVsComp = 1
    CompVsDefender = 2
    CompVsComp = 3

##############################################################################################################

@dataclass(slots=True)
class Unit:
    player: Player = Player.Attacker
    type: UnitType = UnitType.Program
    health : int = 9
    # class variable: damage table for units (based on the unit type constants in order)
    damage_table : ClassVar[list[list[int]]] = [
        [3,3,3,3,1], # AI
        [1,1,6,1,1], # Tech
        [9,6,1,6,1], # Virus
        [3,3,3,3,1], # Program
        [1,1,1,1,1], # Firewall
    ]
    # class variable: repair table for units (based on the unit type constants in order)
    repair_table : ClassVar[list[list[int]]] = [
        [0,1,1,0,0], # AI
        [3,0,0,3,3], # Tech
        [0,0,0,0,0], # Virus
        [0,0,0,0,0], # Program
        [0,0,0,0,0], # Firewall
    ]

    def is_alive(self) -> bool:
        """Are we alive ?"""
        return self.health > 0

    def mod_health(self, health_delta : int):
        """Modify this unit's health by delta amount."""
        self.health += health_delta
        if self.health < 0:
            self.health = 0
        elif self.health > 9:
            self.health = 9

    def to_string(self) -> str:
        """Text representation of this unit."""
        p = self.player.name.lower()[0]
        t = self.type.name.upper()[0]
        return f"{p}{t}{self.health}"

    def __str__(self) -> str:
        """Text representation of this unit."""
        return self.to_string()

    def damage_amount(self, target: Unit) -> int:
        """How much can this unit damage another unit."""
        amount = self.damage_table[self.type.value][target.type.value]
        if target.health - amount < 0:
            return target.health
        return amount

    def repair_amount(self, target: Unit) -> int:
        """How much can this unit repair another unit."""
        amount = self.repair_table[self.type.value][target.type.value]
        if target.health + amount > 9:
            return 9 - target.health
        return amount

##############################################################################################################

@dataclass(slots=True)
class Coord:
    """Representation of a game cell coordinate (row, col)."""
    row : int = 0
    col : int = 0

    def col_string(self) -> str:
        """Text representation of this Coord's column."""
        coord_char = '?'
        if self.col < 16:
                coord_char = "0123456789abcdef"[self.col]
        return str(coord_char)

    def row_string(self) -> str:
        """Text representation of this Coord's row."""
        coord_char = '?'
        if self.row < 26:
                coord_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[self.row]
        return str(coord_char)

    def to_string(self) -> str:
        """Text representation of this Coord."""
        return self.row_string()+self.col_string()

    def __str__(self) -> str:
        """Text representation of this Coord."""
        return self.to_string()

    def clone(self) -> Coord:
        """Clone a Coord."""
        return copy.copy(self)

    def iter_range(self, dist: int) -> Iterable[Coord]:
        """Iterates over Coords inside a rectangle centered on our Coord."""
        for row in range(self.row-dist,self.row+1+dist):
            for col in range(self.col-dist,self.col+1+dist):
                yield Coord(row,col)

    def iter_adjacent(self) -> Iterable[Coord]:
        """Iterates over adjacent Coords."""
        yield Coord(self.row-1,self.col)
        yield Coord(self.row,self.col-1)
        yield Coord(self.row+1,self.col)
        yield Coord(self.row,self.col+1)

    @classmethod
    def from_string(cls, s : str) -> Coord | None:
        """Create a Coord from a string. ex: D2."""
        s = s.strip()
        for sep in " ,.:;-_":
                s = s.replace(sep, "")
        if (len(s) == 2):
            coord = Coord()
            coord.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coord.col = "0123456789abcdef".find(s[1:2].lower())
            return coord
        else:
            return None

##############################################################################################################

@dataclass(slots=True)
class CoordPair:
    """Representation of a game move or a rectangular area via 2 Coords."""
    src : Coord = field(default_factory=Coord)
    dst : Coord = field(default_factory=Coord)

    def to_string(self) -> str:
        """Text representation of a CoordPair."""
        return self.src.to_string()+" "+self.dst.to_string()

    def __str__(self) -> str:
        """Text representation of a CoordPair."""
        return self.to_string()

    def clone(self) -> CoordPair:
        """Clones a CoordPair."""
        return copy.copy(self)

    def iter_rectangle(self) -> Iterable[Coord]:
        """Iterates over cells of a rectangular area."""
        for row in range(self.src.row,self.dst.row+1):
            for col in range(self.src.col,self.dst.col+1):
                yield Coord(row,col)

    @classmethod
    def from_quad(cls, row0: int, col0: int, row1: int, col1: int) -> CoordPair:
        """Create a CoordPair from 4 integers."""
        return CoordPair(Coord(row0,col0),Coord(row1,col1))

    @classmethod
    def from_dim(cls, dim: int) -> CoordPair:
        """Create a CoordPair based on a dim-sized rectangle."""
        return CoordPair(Coord(0,0),Coord(dim-1,dim-1))

    @classmethod
    def from_string(cls, s : str) -> CoordPair | None:
        """Create a CoordPair from a string. ex: A3 B2"""
        s = s.strip()
        for sep in " ,.:;-_":
                s = s.replace(sep, "")
        if (len(s) == 4):
            coords = CoordPair()
            coords.src.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[0:1].upper())
            coords.src.col = "0123456789abcdef".find(s[1:2].lower())
            coords.dst.row = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".find(s[2:3].upper())
            coords.dst.col = "0123456789abcdef".find(s[3:4].lower())
            return coords
        else:
            return None

##############################################################################################################

@dataclass(slots=True)
class Options:
    """Representation of the game options."""
    dim: int = 5
    max_depth : int | None = 4
    min_depth : int | None = 2
    max_time : float | None = 5.0
    game_type : GameType = GameType.AttackerVsDefender
    alpha_beta : bool = True
    max_turns : int | None = 100
    randomize_moves : bool = True
    broker : str | None = None
    attacker_heuristic : int | None = 0
    defender_heuristic : int | None = 0

##############################################################################################################

@dataclass(slots=True)
class Stats:
    """Representation of the global game statistics."""
    evaluations_per_depth : dict[int,int] = field(default_factory=dict)
    total_seconds: float = 0.0
    branching_factor_tuple = [0,0]

##############################################################################################################

@dataclass(slots=True)
class Game:
    """Representation of the game state."""
    board: list[list[Unit | None]] = field(default_factory=list)
    next_player: Player = Player.Attacker
    turns_played : int = 0
    options: Options = field(default_factory=Options)
    stats: Stats = field(default_factory=Stats)
    _attacker_has_ai : bool = True
    _defender_has_ai : bool = True
    # gf: TextIOWrapper

    def __post_init__(self):
        """Automatically called after class init to set up the default board state."""
        dim = self.options.dim
        self.board = [[None for _ in range(dim)] for _ in range(dim)]
        md = dim-1
        self.set(Coord(0,0),Unit(player=Player.Defender,type=UnitType.AI))
        self.set(Coord(1,0),Unit(player=Player.Defender,type=UnitType.Tech))
        self.set(Coord(0,1),Unit(player=Player.Defender,type=UnitType.Tech))
        self.set(Coord(2,0),Unit(player=Player.Defender,type=UnitType.Firewall))
        self.set(Coord(0,2),Unit(player=Player.Defender,type=UnitType.Firewall))
        self.set(Coord(1,1),Unit(player=Player.Defender,type=UnitType.Program))
        self.set(Coord(md,md),Unit(player=Player.Attacker,type=UnitType.AI))
        self.set(Coord(md-1,md),Unit(player=Player.Attacker,type=UnitType.Virus))
        self.set(Coord(md,md-1),Unit(player=Player.Attacker,type=UnitType.Virus))
        self.set(Coord(md-2,md),Unit(player=Player.Attacker,type=UnitType.Program))
        self.set(Coord(md,md-2),Unit(player=Player.Attacker,type=UnitType.Program))
        self.set(Coord(md-1,md-1),Unit(player=Player.Attacker,type=UnitType.Firewall))

    def clone(self) -> Game:
        """Make a new copy of a game for minimax recursion.

        Shallow copy of everything except the board (options and stats are shared).
        """
        new = copy.copy(self)
        new.board = copy.deepcopy(self.board)
        return new

    def is_empty(self, coord : Coord) -> bool:
        """Check if contents of a board cell of the game at Coord is empty (must be valid coord)."""
        return self.board[coord.row][coord.col] is None

    def get(self, coord : Coord) -> Unit | None:
        """Get contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            return self.board[coord.row][coord.col]
        else:
            return None

    def set(self, coord : Coord, unit : Unit | None):
        """Set contents of a board cell of the game at Coord."""
        if self.is_valid_coord(coord):
            self.board[coord.row][coord.col] = unit

    def remove_dead(self, coord: Coord):
        """Remove unit at Coord if dead."""
        unit = self.get(coord)
        if unit is not None and not unit.is_alive():
            self.set(coord,None)
            if unit.type == UnitType.AI:
                if unit.player == Player.Attacker:
                    self._attacker_has_ai = False
                else:
                    self._defender_has_ai = False

    def mod_health(self, coord : Coord, health_delta : int):
        """Modify health of unit at Coord (positive or negative delta)."""
        target = self.get(coord)
        if target is not None:
            target.mod_health(health_delta)
            self.remove_dead(coord)

    def is_valid_move(self, coords : CoordPair) -> bool:
        """Validate a move expressed as a CoordPair. TODO: WRITE MISSING CODE!!!"""
        if not self.is_valid_coord(coords.src) or not self.is_valid_coord(coords.dst):
            return False
        unit = self.get(coords.src)
        if unit is None or unit.player != self.next_player:
            return False
        if self.get(coords.dst) is None:
            if unit.type in [UnitType.Virus, UnitType.Tech]:
                return True
            else:
                for adj_coord in Coord.iter_adjacent(coords.src): #check if unit is not engaged in combat
                        if self.get(adj_coord) is not None and self.get(adj_coord).player != self.get(coords.src).player:
                            return False
                if unit.player == Player.Attacker:
                    if (coords.dst.row == (coords.src.row-1)) or (coords.dst.col == (coords.src.col-1)):
                        return True
                    else:
                        return False
                else:
                    if (coords.dst.row == coords.src.row+1) or (coords.dst.col == coords.src.col+1):
                        return True
                    else:
                        return False
        elif ((coords.src.row == coords.dst.row) and (coords.src.col == coords.dst.col)):
            return True
        elif unit.player is self.get(coords.dst).player:
            if unit.type in [UnitType.Virus, UnitType.Firewall, UnitType.Program] or self.get(coords.dst).health == 9:
                return False
            else:
                return True
        return True

    def perform_move(self, coords : CoordPair) -> Tuple[bool,str]:
        """Validate and perform a move expressed as a CoordPair. TODO: WRITE MISSING CODE!!!"""
        if self.is_valid_move(coords):
            if self.get(coords.dst) is None: #move condition
                if self.get(coords.src).type in [UnitType.Virus, UnitType.Tech]:
                    self.set(coords.dst,self.get(coords.src))
                    self.set(coords.src,None)
                    return (True,"Move from " + str(coords.src) + " to " + str(coords.dst) + "\n\n")
                elif self.get(coords.src).type in [UnitType.AI, UnitType.Firewall, UnitType.Program]:
                    #check no longer needed as is_valid_move performs it
                    #for adj_coord in Coord.iter_adjacent(coords.src): #check if unit is not engaged in combat
                    #    if self.get(adj_coord) is not None and self.get(adj_coord).player != self.get(coords.src).player:
                    #        return (False, "invalid move")
                    if self.get(coords.src).player == Player.Attacker: #if Attacker moves AI, Firewall, or Program, check if going up or left before confirming the move
                        if (coords.dst.row == (coords.src.row-1)) or (coords.dst.col == (coords.src.col-1)):
                            self.set(coords.dst,self.get(coords.src))
                            self.set(coords.src,None)
                            return (True,"Move from " + str(coords.src) + " to " + str(coords.dst) + "\n\n")
                        else:
                            return (False,"invalid move")
                    else: #if Defender moves AI, Firewall, or Program, check if going down or right before confirming the move
                        if (coords.dst.row == coords.src.row+1) or (coords.dst.col == coords.src.col+1):
                            self.set(coords.dst,self.get(coords.src))
                            self.set(coords.src,None)
                            return (True,"Move from " + str(coords.src) + " to " + str(coords.dst) + "\n\n")
                        else:
                            return (False,"invalid move")
            elif ((coords.src.row == coords.dst.row) and (coords.src.col == coords.dst.col)): #self-destruct condition
                self.mod_health(coords.src, -9)
                for i in coords.src.iter_range(1):
                    if self.get(i) is not None:
                        self.mod_health(i, -2)
                return (True,"Unit on " + str(coords.src) + " self-destructed \n\n")
            ###############      //   Nayeem         ##########################
            else:
                unit_src = self.get(coords.src)
                unit_dst = self.get(coords.dst)
                if unit_dst is None:
                    return (False,"")
                elif unit_src.player is unit_dst.player: ## Friendly unit -> repair
                    if unit_src.type in [UnitType.Virus, UnitType.Firewall, UnitType.Program]:
                        return (False, "")
                    else:
                        if (unit_dst.health == 9):
                            return (False, "")
                        else:
                            self.mod_health(coords.dst, (unit_src.repair_amount(unit_dst)))
                            return (True,"Unit on " + str(coords.src) + " repairs unit on " + str(coords.dst) + "\n\n")
                elif unit_src.player is not unit_dst.player: ## Adversarial unit -> damage
                    self.mod_health(coords.src, -(unit_dst.damage_amount(unit_src)))
                    self.mod_health(coords.dst, -(unit_src.damage_amount(unit_dst)))
                    return (True,"Unit on " + str(coords.src) + " attacks unit on " + str(coords.dst) + "\n\n")
            ###############         Nayeem         ##########################
        return (False,"invalid move")

    def next_turn(self):
        """Transitions game to the next turn."""
        self.next_player = self.next_player.next()
        self.turns_played += 1

    def to_string(self) -> str:
        """Pretty text representation of the game."""
        dim = self.options.dim
        output = ""
        output += f"Next player: {self.next_player.name}\n"
        output += f"Turns played: {self.turns_played}\n"
        coord = Coord()
        output += "\n   "
        for col in range(dim):
            coord.col = col
            label = coord.col_string()
            output += f"{label:^3} "
        output += "\n"
        for row in range(dim):
            coord.row = row
            label = coord.row_string()
            output += f"{label}: "
            for col in range(dim):
                coord.col = col
                unit = self.get(coord)
                if unit is None:
                    output += " .  "
                else:
                    output += f"{str(unit):^3} "
            output += "\n"
        return output

    def __str__(self) -> str:
        """Default string representation of a game."""
        return self.to_string()

    def is_valid_coord(self, coord: Coord) -> bool:
        """Check if a Coord is valid within out board dimensions."""
        dim = self.options.dim
        if coord.row < 0 or coord.row >= dim or coord.col < 0 or coord.col >= dim:
            return False
        return True

    def read_move(self) -> CoordPair:
        """Read a move from keyboard and return as a CoordPair."""
        while True:
            s = input(F'Player {self.next_player.name}, enter your move: ')
            coords = CoordPair.from_string(s)
            if coords is not None and self.is_valid_coord(coords.src) and self.is_valid_coord(coords.dst):
                return coords
            else:
                print('Invalid coordinates! Try again.')

    def human_turn(self):
        """Human player plays a move (or get via broker)."""
        if self.options.broker is not None:
            print("Getting next move with auto-retry from game broker...")
            while True:
                mv = self.get_move_from_broker()
                if mv is not None:
                    (success,result) = self.perform_move(mv)
                    print(f"Broker {self.next_player.name}: ",end='')
                    print(result)
                    if success:
                        self.next_turn()
                        break
                sleep(0.1)
        else:
            while True:
                mv = self.read_move()
                (success,result) = self.perform_move(mv)
                if success:
                    print(f"Player {self.next_player.name}: ",end='')
                    print(result)
                    gf.write(result)    #added a write to the file here to indicate the played result
                    self.next_turn()
                    break
                else:
                    print("The move is not valid! Try again.")
                    gf.write("The move is not valid! Try again.\n") #added a write to the file to indicate that the played move is invalid

    def heuristic(self) -> int:
        player1_units = self.player_units(self.next_player)
        player2_units = self.player_units(self.next_player.next())
        heuristic_id = 0
        heuristic_value = 0
        if self.next_player == Player.Attacker:
            heuristic_id = Options.attacker_heuristic
        else:
            heuristic_id = Options.defender_heuristic

        if heuristic_id == 0: #heuristic given in the handout
            for i in player1_units:
                if i[1].type == UnitType.AI:
                    heuristic_value += 9999
                else:
                    heuristic_value += 3
            for i in player2_units:
                if i[1].type == UnitType.AI:
                    heuristic_value += -9999
                else:
                    heuristic_value += -3
            return heuristic_value
        elif heuristic_id == 1: #more aggressive heuristic, Virus and Tech pieces are valued more than Program and Firewall and capturing the opponent's pieces have more impact than losing a piece
            for i in player1_units:
                if i[1].type == UnitType.AI:
                    heuristic_value += 9999
                elif i[1].type in [UnitType.Virus, UnitType.Tech]:
                    heuristic_value += 6
                else:
                    heuristic_value += 3
            for i in player2_units:
                if i[1].type == UnitType.AI:
                    heuristic_value += -9999
                elif i[1].type in [UnitType.Virus, UnitType.Tech]:
                    heuristic_value += -10
                else:
                    heuristic_value += -6
            return heuristic_value
        else: #
            for i in player1_units:
                if i[1].type == UnitType.AI:
                    heuristic_value += 9999*i[1].health
                elif i[1].type in [UnitType.Virus, UnitType.Tech]:
                    heuristic_value += 6*i[1].health
                else:
                    heuristic_value +=3*i[1].health
            for i in player2_units:
                if i[1].type == UnitType.AI:
                    heuristic_value += -9999*i[1].health
                elif i[1].type in [UnitType.Virus, UnitType.Tech]:
                    heuristic_value += -10*i[1].health
                else:
                    heuristic_value += -6*i[1].health
            return heuristic_value

    def minimax (self,
                 depth: int,
                 maximizing: bool,
                 start_time: datetime):
        children = list(self.move_candidates())
        self.stats.branching_factor_tuple[0] += children.__len__
        self.stats.branching_factor_tuple[1] += 1
        elapsed_time = (datetime.now() - start_time).total_seconds()
        if depth == self.options.max_depth or children == None or (elapsed_time >= 0.95 * self.options.max_time):
            return (self.heuristic(),None)

        if maximizing:
            maxScore = (-10000000, None)
            for child in children:
                temp = self.clone()
                temp.perform_move(child)
                minimaxScore = temp.minimax(depth+1, False, start_time)
                if maxScore[0] < minimaxScore[0]:
                    maxScore = (minimaxScore[0], child)
            return maxScore
        else:
            minScore = (10000000, None)
            for child in children:
                otherTemp = self.clone()
                otherTemp.perform_move(child)
                minimaxScore = otherTemp.minimax(depth+1, True, start_time)
                if minScore[0] > minimaxScore[0]:
                    minScore = (minimaxScore[0], child)
            return minScore

    def alphabeta(self,
                  depth: int,
                  maximizing: bool,
                  start_time: datetime,
                  alpha = -1000000,
                  beta = 1000000):

        children = list(self.move_candidates())
        self.stats.branching_factor_tuple[0] = self.stats.branching_factor_tuple[0] + len(children)        # added this (same code from minimax) to compute the branching factor
        self.stats.branching_factor_tuple[1] = self.stats.branching_factor_tuple[1] + 1
        elapsed_time = (datetime.now() - start_time).total_seconds()


        if depth == self.options.max_depth or children == None or (elapsed_time >= 0.95 * self.options.max_time):
            return (self.heuristic(),None)
        if maximizing:
            maxScore = (-10000000, None)
            for child in children:
                temp = self.clone()
                temp.perform_move(child)
                minimaxScore = temp.alphabeta(depth-1, False, start_time, alpha, beta)
                if maxScore[0] < minimaxScore[0]:
                    maxScore = (minimaxScore[0], child)
                    alpha = max(alpha, maxScore[0])
                if  beta <= alpha:
                    break
            return maxScore
        else:
            minScore = (10000000, None)
            for child in children:
                otherTemp = self.clone()
                otherTemp.perform_move(child)
                minimaxScore = otherTemp.alphabeta(depth-1, True, alpha, beta)
                if minScore[0] > minimaxScore[0]:
                    minScore = (minimaxScore[0], child)
                    beta = min(beta, minScore[0])
                if beta <= alpha:
                    break
            return minScore

    def computer_turn(self) -> CoordPair | None:
        """Computer plays a move."""
        mv = self.suggest_move()
        if mv is not None:
            (success,result) = self.perform_move(mv)
            if success:
                print(f"Computer {self.next_player.name}: ",end='')
                print(result)
                self.next_turn()
        return mv

    def player_units(self, player: Player) -> Iterable[Tuple[Coord,Unit]]:
        """Iterates over all units belonging to a player."""
        for coord in CoordPair.from_dim(self.options.dim).iter_rectangle():
            unit = self.get(coord)
            if unit is not None and unit.player == player:
                yield (coord,unit)

    def is_finished(self) -> bool:
        """Check if the game is over."""
        return self.has_winner() is not None

    def has_winner(self) -> Player | None:
        """Check if the game is over and returns winner"""
        if self.options.max_turns is not None and self.turns_played >= self.options.max_turns:
            return Player.Defender
        if self._attacker_has_ai:
            if self._defender_has_ai:
                return None
            else:
                return Player.Attacker
        return Player.Defender

    def move_candidates(self) -> Iterable[CoordPair]:
        """Generate valid move candidates for the next player."""
        move = CoordPair()
        for (src,_) in self.player_units(self.next_player):
            move.src = src
            for dst in src.iter_adjacent():
                move.dst = dst
                if self.is_valid_move(move):
                    yield move.clone()
            move.dst = src
            yield move.clone()

    def random_move(self) -> Tuple[int, CoordPair | None, float]:
        """Returns a random move."""
        move_candidates = list(self.move_candidates())
        random.shuffle(move_candidates)
        if len(move_candidates) > 0:
            return (0, move_candidates[0], 1)
        else:
            return (0, None, 0)

    def suggest_move(self) -> CoordPair | None:
        """Suggest the next move using minimax alpha beta. TODO: REPLACE RANDOM_MOVE WITH PROPER GAME LOGIC!!!"""
        start_time = datetime.now()
        if Options.alpha_beta:
            (score, move) = self.alphabeta(0, True, start_time)
        else:
            (score, move) = self.minimax(0, True, start_time)
        elapsed_seconds = (datetime.now() - start_time).total_seconds()
        self.stats.total_seconds += elapsed_seconds
        print(f"Elapsed time: {elapsed_seconds:0.1f}s")
        gf.write(f"Elapsed time: {elapsed_seconds:0.1f}s")                                                                              # added file write here
        print(f"Heuristic score: {score} \n")
        gf.write(f"Heuristic score: {score} \n")                                                                                        # added file write here
        total_evals = sum(self.stats.evaluations_per_depth.values())
        print(f"Cumulative evals: {total_evals/1000000}M \n")
        gf.write(f"Cumulative evals: {total_evals/1000000}M \n")                                                                        # added file write here
        print(f"Cumulative evals per depth: ",end='')
        gf.write(f"Cumulative evals per depth: ",end='')                                                                                # added file write here
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            print(f"{k}:{self.stats.evaluations_per_depth[k]} ",end='')
            gf.write(f"{k}:{self.stats.evaluations_per_depth[k]} ",end='')                                                              # added file write here
        print()
        gf.write("\n")                                                                                                                  # added file write here
        print(f"Cumulative % evals per depth: ",end='')
        gf.write(f"Cumulative % evals per depth: ",end='')                                                                              # added file write here
        for k in sorted(self.stats.evaluations_per_depth.keys()):
            print(f"{k}:{self.stats.evaluations_per_depth[k]/total_evals}% ",end='')
            gf.write(f"{k}:{self.stats.evaluations_per_depth[k]/total_evals}% ",end='')                                                 # added file write here
        #if self.stats.total_seconds > 0:
        #    print(f"Eval perf.: {total_evals/self.stats.total_seconds/1000:0.1f}k/s")
        print(f"Average branching factor: {self.stats.branching_factor_tuple[0]/self.stats.branching_factor_tuple[1]} \n",end='')
        gf.write(f"Average branching factor: {self.stats.branching_factor_tuple[0]/self.stats.branching_factor_tuple[1]} \n",end='')    # added file write here
        return move

    def post_move_to_broker(self, move: CoordPair):
        """Send a move to the game broker."""
        if self.options.broker is None:
            return
        data = {
            "from": {"row": move.src.row, "col": move.src.col},
            "to": {"row": move.dst.row, "col": move.dst.col},
            "turn": self.turns_played
        }
        try:
            r = requests.post(self.options.broker, json=data)
            if r.status_code == 200 and r.json()['success'] and r.json()['data'] == data:
                # print(f"Sent move to broker: {move}")
                pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")

    def get_move_from_broker(self) -> CoordPair | None:
        """Get a move from the game broker."""
        if self.options.broker is None:
            return None
        headers = {'Accept': 'application/json'}
        try:
            r = requests.get(self.options.broker, headers=headers)
            if r.status_code == 200 and r.json()['success']:
                data = r.json()['data']
                if data is not None:
                    if data['turn'] == self.turns_played+1:
                        move = CoordPair(
                            Coord(data['from']['row'],data['from']['col']),
                            Coord(data['to']['row'],data['to']['col'])
                        )
                        print(f"Got move from broker: {move}")
                        return move
                    else:
                        # print("Got broker data for wrong turn.")
                        # print(f"Wanted {self.turns_played+1}, got {data['turn']}")
                        pass
                else:
                    # print("Got no data from broker")
                    pass
            else:
                print(f"Broker error: status code: {r.status_code}, response: {r.json()}")
        except Exception as error:
            print(f"Broker error: {error}")
        return None

##############################################################################################################

def main():
    #creating a variable that will hold the "game type value (human vs human, human vs ai, ai vs human, ai vs ai)"
    gt = ""

    # parse command line arguments
    parser = argparse.ArgumentParser(
        prog='ai_wargame',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--max_depth', type=int, help='maximum search depth')
    parser.add_argument('--max_time', type=float, help='maximum search time')
    parser.add_argument('--max_turn', type=int, help='maximum number of turns')
    parser.add_argument('--game_type', type=str, default="manual", help='game type: auto|attacker|defender|manual')
    parser.add_argument('--alpha_beta', type=bool, help='True for alpha-beta or False for minimax for search')
    parser.add_argument('--broker', type=str, help='play via a game broker')
    parser.add_argument('--attack_h', type=int, help='the heuristic the attacker will use if it\'s an AI player, enter int value of 0, 1, or 2')
    parser.add_argument('--defend_h', type=int, help='the heuristic the defender will use if it\'s an AI player, enter int value of 0, 1, or 2')
    args = parser.parse_args()

    # parse the game type
    if args.game_type == "attacker":
        game_type = GameType.AttackerVsComp
        gt = "Player 1 = H & Player 2 = AI\n"
    elif args.game_type == "defender":
        game_type = GameType.CompVsDefender
        gt = "Player 1 = H & Player 2 = AI\n"
    elif args.game_type == "manual":
        game_type = GameType.AttackerVsDefender
        gt = "Player 1 = H & Player 2 = H\n"
    else:
        game_type = GameType.CompVsComp
        gt = "Player 1 = AI & Player 2 = AI\n"


    # set up game options
    options = Options(game_type=game_type)


    # override class defaults via command line options
    if args.max_depth is not None:
        options.max_depth = args.max_depth

    if args.max_turn is not None:
        options.max_turns = args.max_turn

    if args.max_time is not None:
        options.max_time = args.max_time

    if args.alpha_beta is not None:
        options.alpha_beta = args.alpha_beta

    if args.broker is not None:
        options.broker = args.broker

    if args.attack_h is 0 or 1 or 2:
        options.attacker_heuristic = args.attack_h
    if args.defend_h is 0 or 1 or 2:
        options.defender_heuristic = args.defend_h

#adding the variables for the necessary information, then combining them into 1 string. Making the variables since they are used in 2 places.
    filename = "gameTrace-"+(Options.alpha_beta.__str__)+ "-" + str(options.max_time)+ "-" + str(options.max_turns)

    gf.write("==============================\ngameTrace-" + str(options.alpha_beta)+ "-" + str(options.max_time)+ "-" + str(options.max_turns)+ "\n" + "\n" + gt)
    gf.write("Timeout: " + str(options.max_time) + "\nMax number of turns: " + str(options.max_turns) + "\nAlpha-Beta state: " + str(options.alpha_beta) + "\nPlay Mode: " + gt)

    # create a new game
    game = Game(options=options)
    gf.write("\nInitial Board Setup:\n" + str(game))
    # the main game loop
    while True:
        print()
        #CODE ADDED HERE
        print(game)

        winner = game.has_winner()
        if winner is not None:
            print(f"{winner.name} wins!")
            gf.write(winner.name + " wins in " + str(game.turns_played) + " moves!\n")              #Prints the winner of the game
            break
        if game.options.game_type == GameType.AttackerVsDefender:
            game.human_turn()
        elif game.options.game_type == GameType.AttackerVsComp and game.next_player == Player.Attacker:
            game.human_turn()
        elif game.options.game_type == GameType.CompVsDefender and game.next_player == Player.Defender:
            game.human_turn()
        else:
            player = game.next_player
            move = game.computer_turn()
            if move is not None:
                game.post_move_to_broker(move)
            else:
                print("Computer doesn't know what to do!!!")
                gf.write("Computer doesn't know what to do!!!") #added a write to file if the computer doesn't know what to do
                exit(1)
        gf.write(str(game)+"\n")    #writes the current board state to the file
    gf.close()  #added the close command to write to the file


    copyfile("output.txt", f"{filename}.txt")   # creates a new text file with the game trace in it, does a copy of the current game and stores it in a appropriately named file.


##############################################################################################################

if __name__ == '__main__':
    main()
