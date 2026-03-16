'''
    madness.py

    Copyright (c) 2025, Reid Simmons, Carnegie Mellon University
      This software is distributed under the terms of the 
      Simplified BSD License (see ipc/LICENSE.TXT)
'''
import os
import pandas as pd
from utils import read_data, read_tourney_predictions

class ShowBracket:
    def __init__(self, bracket, item_width=10):
        self.bracket = bracket
        self.item_width = item_width

    def printable(self, str, correct, spacer='_'):
        slen = len(str)
        lstr, rstr = ('','') if correct else ('\033[9m', '\033[0m')
        if slen > self.item_width:
            str = str[:self.item_width]
        else:
            slack = self.item_width - slen
            slack_str = spacer*((slack+1)//2)
            str = slack_str[:slack//2] + str + slack_str[:(slack+1)//2]
        return (str if correct else 
                '\u0336'.join(" "+str)[1:] if self.use_unicode else 
                '\033[9m' + str + '\033[0m')

    def add_delim(self, round, line):
        round_delims = {1: [], 2: [2], 3: [2,4,5], 
                       4: range(4,12), 5: range(8,24), 
                       6: list(range(16,31)) + list(range(32,48)),
                       7: range(26,32)}
        posn = 2**round
        return line%posn in round_delims[round]
            
    def team_line(self, round, line):
        return (True if round == 7 and line == 25 else
                line%(2**round) == (2**(round-1)-1))

    def skip(self, round, line, left_side):
        spaces = ' '*(self.item_width-1 if round==1 else self.item_width)
        add_delim = self.add_delim(round, line)
        ldelim = '|' if add_delim else ' '
        if round==7: spaces += '|' if add_delim else ' '
        print(ldelim+spaces if left_side else spaces+ldelim, end="")

    def show_team(self, region, round, slot, left_side):
        ldelim = '' if not left_side or round == 1 else ' ' if round == 7 else '|'
        rdelim = ' ' if round==7 else '' if left_side or round==1 else '|'
        id = self.bracket.get_predicted(region, round, slot)
        team = self.bracket.get_team(id) if id else ''
        correct = self.bracket.get_correct(region, round, slot) != False # Could be unknown
        print(ldelim+self.printable(team, correct)+rdelim, end='')

    def show_correct_team(self, region, round, line, left_side):
        nline = line + 1
        nregion = region if round < 5 or line != 33 else 'W' if left_side else 'Y'
        nslot = 1+nline%32//(2**round)
        if (line < 62 and self.team_line(round, nline) and
            self.bracket.get_correct(nregion, round, nslot) == False and
            self.bracket.get_actual(nregion, round, nslot) != None):
            id = self.bracket.get_actual(nregion, round, nslot)
            add_delim = self.add_delim(round, line)
            ldelim = '|' if add_delim and left_side else ' ' if left_side and round > 1 else ''
            rdelim = '|' if add_delim and not left_side else ' ' if round==7 or (not left_side and round == 2) else ''
            print(ldelim+self.printable(self.bracket.get_team(id), True, ' ')+rdelim, end='')
            return True
        return False
        
    def show_half_line(self, region, rounds, line, left_side):
        for round in rounds:
            slot = 1+line%32//(2**round)
            if self.team_line(round, line):
                self.show_team(region, round, slot, left_side)
            elif not self.show_correct_team(region, round, line, left_side):
                self.skip(round, line, left_side)
        
    def show_line(self, line):
        # For strikethrough to work correctly with unicode first character needs to be space
        if self.use_unicode: print(end=' ')
        self.show_half_line('W' if line < 32 else 'X', range(1,8), line, True)
        self.show_half_line('Y' if line < 32 else 'Z', range(6,0, -1), line, False)
        print()

    def show_playins(self): # Show play-in games
        spaces = " "*(self.item_width*5-1)
        playin_cells = self.bracket.bracket.loc[self.bracket.bracket['playin'] == 'a']
        for i in range(len(playin_cells)):
            region = playin_cells.iloc[i]['region']
            slot = playin_cells.iloc[i]['slot']
            playins = self.bracket.get_cell(region, 0, slot)['pred']
            print(spaces, "Play-in game %s%s: %s vs %s" 
                  %(region, slot, self.bracket.get_team(playins.iloc[0]),
                    self.bracket.get_team(playins.iloc[1])))

    def show(self, use_unicode=False):
        self.use_unicode = use_unicode
        for line in range(0, 63):
            self.show_line(line)
        self.show_playins()

class Bracket:
    def __init__(self, season, which='M'):
        self.which = which
        self.season = season
        self.regions = ['W', 'X', 'Y', 'Z']
        # Who plays in each slot
        self.slot_seeds = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
        self.teams = read_data("Teams", self.which)
        self.create()

    def get_team(self, id):
        return self.teams[self.teams['TeamID'] == id]['TeamName'].iloc[0]
        try:
            return self.teams[self.teams['TeamID'] == id]['TeamName'].iloc[0]
        except:
            print("ERROR: id %s not found" %id)
            return ''
    
    def get_id(self, teamname):
        try:
            return self.teams[self.teams['TeamName'] == teamname]['TeamID'].iloc[0]
        except:
            print("ERROR: team %s not found" %teamname)
            return 0

    def get_slot(self, seed):
        return self.slot_seeds.index(seed)+1

    def cell_index(self, region, round, slot):
        b = self.bracket
        return ((b['region'] == region) & (b['round'] == round) & 
                (b['slot'] == slot))
        
    def get_cell(self, region, round, slot):
        return self.bracket.loc[self.cell_index(region, round, slot)]
    
    def get_cell_column(self, region, round, slot, column):
        return self.get_cell(region, round, slot)[column].iloc[0]
    def set_cell_column(self, region, round, slot, column, value):
        self.bracket.loc[self.cell_index(region, round, slot), column] = value

    def create(self):
        cells = [[region, round, slot+1] for region in self.regions
                 for round in range(1,6) for slot in range(2**(5-round))]
        cells += [['W', 6, 1], ['Y', 6, 1], ['W', 7, 1]]
        self.bracket = pd.DataFrame(cells, columns=['region', 'round', 'slot'])
        self.bracket[['playin', 'pred', 'actual', 'correct']] = None

    def seed(self):
        seeds_data = read_data("NCAATourneySeeds", self.which)
        seedings = seeds_data[seeds_data['Season'] == self.season]
        if len(seedings) == 0:
            # If official seeds for this season are not available in data/,
            # use known 2026 men's matchups if provided; otherwise fall back.
            if self.which == 'M' and self.season == 2026:
                self.seed_2026_mens_espn()
            else:
                self.seed_synthetic()
            return
        for i in range(len(seedings)):
            id = seedings.iloc[i]['TeamID']
            region_seed = seedings.iloc[i]['Seed']
            region, seed = region_seed[0], int(region_seed[1:3])
            if len(region_seed) == 3:
                slot = self.get_slot(seed)
                self.set_predicted(region, 1, slot, id)
                # Seedings are always correct
                self.set_actual(region, 1, slot, id)
            else: # Add a playin
                self.bracket.loc[len(self.bracket)] = [region, 0, seed, 
                                                       region_seed[3:4], id, id, None]

    def _normalize_team_key(self, s: str) -> str:
        s = s.lower()
        out = []
        for ch in s:
            if ch.isalnum():
                out.append(ch)
        return "".join(out)

    def _team_id_from_name(self, name: str) -> int:
        """
        Resolve a human-readable name to TeamID, using normalization and a small alias table.
        """
        aliases = {
            "ohio state": "Ohio St",
            "uconn": "Connecticut",
            "ca baptist": "Cal Baptist",
            "california baptist": "Cal Baptist",
            "long island": "LIU Brooklyn",
            "utah state": "Utah St",
            "kennesaw st": "Kennesaw",
            "miami": "Miami FL",
            "queens": "Queens NC",
            "mcneese": "McNeese St",
            "saint mary's": "St Mary's CA",
            "saint marys": "St Mary's CA",
            "saint louis": "St Louis",
            "iowa state": "Iowa St",
        }
        n = aliases.get(name.lower(), name)
        teams = self.teams
        key_to_id = {self._normalize_team_key(tn): int(tid) for tid, tn in zip(teams["TeamID"], teams["TeamName"])}
        k = self._normalize_team_key(n)
        if k in key_to_id:
            return key_to_id[k]
        raise Exception(f"Could not resolve team name '{name}' (normalized '{k}')")

    def _avg_margin_by_team(self):
        results = read_data("RegularSeasonDetailedResults", self.which)
        results = results[results["Season"] == self.season].copy()
        if results.empty:
            raise Exception(f"No regular season data for season {self.season} ({self.which})")
        results["Wmargin"] = results["WScore"] - results["LScore"]
        w = results[["WTeamID", "Wmargin"]].rename(columns={"WTeamID": "TeamID", "Wmargin": "margin"})
        l = results[["LTeamID", "Wmargin"]].rename(columns={"LTeamID": "TeamID", "Wmargin": "margin"})
        l["margin"] = -l["margin"]
        return pd.concat([w, l], ignore_index=True).groupby("TeamID")["margin"].mean().sort_values(ascending=False)

    def seed_2026_mens_espn(self):
        """
        Seed the 2026 men's bracket from the ESPN-provided matchups in the prompt.

        Regions mapping used by this code:
        - W: East, X: West, Y: South, Z: Midwest

        Any TBD teams are filled from remaining teams by 2026 average margin.
        """
        # Region letters used by the bracket printer
        EAST, WEST, SOUTH, MIDWEST = "W", "X", "Y", "Z"

        # Seed slots: (region, seed) -> team name (None means TBD)
        seeds = {
            # EAST
            (EAST, 1): "Duke",
            (EAST, 16): "Siena",
            (EAST, 8): "Ohio State",
            (EAST, 9): "TCU",
            (EAST, 5): "St John's",
            (EAST, 12): "Northern Iowa",
            (EAST, 4): "Kansas",
            (EAST, 13): "CA Baptist",
            (EAST, 6): "Louisville",
            (EAST, 11): "South Florida",
            (EAST, 3): "Michigan St",
            (EAST, 14): "N Dakota St",
            (EAST, 7): "UCLA",
            (EAST, 10): "UCF",
            (EAST, 2): "UConn",
            (EAST, 15): "Furman",
            # WEST
            (WEST, 1): "Arizona",
            (WEST, 16): "Long Island",
            (WEST, 8): "Villanova",
            (WEST, 9): "Utah State",
            (WEST, 5): "Wisconsin",
            (WEST, 12): "High Point",
            (WEST, 4): "Arkansas",
            (WEST, 13): "Hawai'i",
            (WEST, 6): "BYU",
            (WEST, 11): None,  # play-in: TEX/NCSU
            (WEST, 3): "Gonzaga",
            (WEST, 14): "Kennesaw St",
            (WEST, 7): "Miami",
            (WEST, 10): "Missouri",
            (WEST, 2): "Purdue",
            (WEST, 15): "Queens",
            # SOUTH
            (SOUTH, 1): "Florida",
            (SOUTH, 16): None,  # play-in: PV/LEH
            (SOUTH, 8): "Clemson",
            (SOUTH, 9): "Iowa",
            (SOUTH, 5): "Vanderbilt",
            (SOUTH, 12): "McNeese",
            (SOUTH, 4): "Nebraska",
            (SOUTH, 13): "Troy",
            (SOUTH, 6): "North Carolina",
            (SOUTH, 11): "VCU",
            (SOUTH, 3): "Illinois",
            (SOUTH, 14): "Penn",
            (SOUTH, 7): "Saint Mary's",
            (SOUTH, 10): "Texas A&M",
            (SOUTH, 2): "Houston",
            (SOUTH, 15): "Idaho",
            # MIDWEST
            (MIDWEST, 1): "Michigan",
            (MIDWEST, 16): None,  # play-in: UMBC/HOW
            (MIDWEST, 8): "Georgia",
            (MIDWEST, 9): "Saint Louis",
            (MIDWEST, 5): "Texas Tech",
            (MIDWEST, 12): "Akron",
            (MIDWEST, 4): "Alabama",
            (MIDWEST, 13): "Hofstra",
            (MIDWEST, 6): "Tennessee",
            (MIDWEST, 11): None,  # play-in: M-OH/SMU
            (MIDWEST, 3): "Virginia",
            (MIDWEST, 14): "Wright St",
            (MIDWEST, 7): "Kentucky",
            (MIDWEST, 10): "Santa Clara",
            (MIDWEST, 2): "Iowa State",
            (MIDWEST, 15): "Tennessee St",
        }

        # NOTE: ESPN bracket has the 11-seed play-ins in these regions:
        # - West: Texas vs NC State (winner plays BYU)
        # - Midwest: SMU vs Miami OH (winner plays Tennessee)
        playins = {
            (WEST, 11): ("NC State", "Texas"),
            (MIDWEST, 11): ("SMU", "Miami OH"),
            (SOUTH, 16): ("Lehigh", "Prairie View"),
            (MIDWEST, 16): ("Howard", "UMBC"),
        }

        # Resolve all explicitly named teams
        chosen_ids = set()
        resolved = {}
        for (region, seed), name in seeds.items():
            if name is None:
                continue
            tid = self._team_id_from_name(name)
            resolved[(region, seed)] = tid
            chosen_ids.add(tid)

        # Fill any remaining TBD slots with best remaining teams by margin
        margins = self._avg_margin_by_team()
        remaining = [int(t) for t in margins.index.tolist() if int(t) not in chosen_ids]
        for (region, seed), name in list(seeds.items()):
            if name is None and (region, seed) not in playins:
                tid = remaining.pop(0)
                resolved[(region, seed)] = tid
                chosen_ids.add(tid)

        # Seed all non-playin slots
        for (region, seed), tid in resolved.items():
            slot = self.get_slot(seed)
            self.set_predicted(region, 1, slot, tid)
            self.set_actual(region, 1, slot, tid)

        # Add play-in rows (two teams per (region,seed))
        for (region, seed), (n1, n2) in playins.items():
            t1 = self._team_id_from_name(n1)
            t2 = self._team_id_from_name(n2)
            for code, tid in zip(["a", "b"], [t1, t2]):
                self.bracket.loc[len(self.bracket)] = [region, 0, seed, code, tid, tid, None]

    def seed_synthetic(self):
        """
        Create a 68-team bracket seeding without an official Seeds file.
        Picks top 68 teams by a simple 2026 regular-season margin metric, then
        assigns them into standard seed slots across regions, with four play-ins.
        """
        results = read_data("RegularSeasonDetailedResults", self.which)
        results = results[results['Season'] == self.season].copy()
        if results.empty:
            raise Exception(f"No regular season data for season {self.season} ({self.which})")

        # Average scoring margin per team
        results["Wmargin"] = results["WScore"] - results["LScore"]
        w = results[["WTeamID", "Wmargin"]].rename(columns={"WTeamID": "TeamID", "Wmargin": "margin"})
        l = results[["LTeamID", "Wmargin"]].rename(columns={"LTeamID": "TeamID", "Wmargin": "margin"})
        l["margin"] = -l["margin"]
        margins = pd.concat([w, l], ignore_index=True).groupby("TeamID")["margin"].mean().sort_values(ascending=False)

        field = margins.index.tolist()[:68]

        # Seed list: 1..16 per region, with four play-in seeds (we'll use 11 and 16 like typical)
        regions = self.regions
        # Assign top 64 directly, last 4 into play-in slots (2 games) by giving them duplicate seeds
        direct = field[:64]
        playin = field[64:]

        # Fill direct teams into regions/seeds in order
        idx = 0
        for seed in range(1, 17):
            for region in regions:
                if idx >= len(direct):
                    break
                team_id = direct[idx]
                idx += 1
                slot = self.get_slot(seed)
                self.set_predicted(region, 1, slot, team_id)
                self.set_actual(region, 1, slot, team_id)

        # Add four play-in teams: two games at seed 16 in regions W and Y
        # (This is a simple placeholder; real play-in assignment differs year to year.)
        if len(playin) == 4:
            for region, seed, pair in [("W", 16, playin[:2]), ("Y", 16, playin[2:])]:
                for code, tid in zip(["a", "b"], pair):
                    self.bracket.loc[len(self.bracket)] = [region, 0, seed, code, tid, tid, None]
                
    def next_region(self, cur_region, cur_round):
        return (cur_region if cur_round < 5 else 
                'W' if cur_round == 6 or cur_region in ['W', 'X'] else 'Y')

    def get_predicted(self, region, round, slot):
        return self.get_cell_column(region, round, slot, 'pred')
    def set_actual(self, region, round, slot, actual):
        self.set_cell_column(region, round, slot, 'actual', actual)
     
    def get_actual(self, region, round, slot):
        return self.get_cell_column(region, round, slot, 'actual')
    def set_predicted(self, region, round, slot, pred):
        self.set_cell_column(region, round, slot, 'pred', pred)
      
    def is_correct(self, region, round, slot, winner):
        return winner == self.get_predicted(region, round, slot)
    def get_correct(self, region, round, slot):
        return self.get_cell_column(region, round, slot, 'correct')
    def set_correct(self, region, round, slot, is_correct):
        self.set_cell_column(region, round, slot, 'correct', is_correct)

    def get_winner(self, id1, id2, predictions):
        matchup = predictions.loc[(predictions['WTeamID'] == id1) & (predictions['LTeamID'] == id2)]
        if not matchup.empty:
            return id1
        else:
            matchup = predictions.loc[(predictions['WTeamID'] == id2) & (predictions['LTeamID'] == id1)]
            if not matchup.empty:
                return id2
            else:
                raise Exception("No matchup found for %s vs %s" %(id1, id2))

    def build_basic_predictions(self):
        seeds_data = read_data("NCAATourneySeeds", self.which)
        seeds = seeds_data[seeds_data['Season'] == self.season][['TeamID', 'Seed']]
        if len(seeds) == 0:
            raise Exception("No seed data found for season %s" % self.season)
        # Extract numeric seed (characters 1-2 of the seed string, after region letter)
        seeds['SeedNum'] = seeds['Seed'].str[1:3].astype(int)
        seed_map = dict(zip(seeds['TeamID'], seeds['SeedNum']))
        team_ids = list(seed_map.keys())
        rows = []
        for i in range(len(team_ids)):
            for j in range(i+1, len(team_ids)):
                id1, id2 = team_ids[i], team_ids[j]
                s1, s2 = seed_map[id1], seed_map[id2]
                if s1 < s2 or (s1 == s2 and id1 < id2):
                    winner, loser = id1, id2
                else:
                    winner, loser = id2, id1
                rows.append((winner, loser))
        return pd.DataFrame(rows, columns=['WTeamID', 'LTeamID'])

    def fill(self):
        try:
            predictions = read_tourney_predictions(self.which)
        except FileNotFoundError:
            # If no predictions file exists, fall back to a simple, seed-based model.
            # (madness.py's __main__ can optionally generate ML predictions.)
            predictions = self.build_basic_predictions()
        # Start with playin games, then move from round to round
        playin_cells = self.bracket.loc[self.bracket['playin'] == 'a']
        for i in range(len(playin_cells)):
            region = playin_cells.iloc[i]['region']
            seed = playin_cells.iloc[i]['slot']
            playins = self.get_cell(region, 0, seed)['pred']
            self.set_predicted(region, 1, self.get_slot(seed), 
                               self.get_winner(playins.iloc[0], playins.iloc[1], predictions))
            # Need to handle playins specially
            self.set_correct(region, 1,  self.get_slot(seed), 'waiting')
        # Next do the regional rounds
        for region in self.regions:
            for round in range(1,5):
                for slot in range(1, 2**(5-round), 2):
                    winner = self.get_winner(self.get_predicted(region, round, slot),
                                             self.get_predicted(region, round, slot+1), predictions)
                    self.set_predicted(region, round+1, (slot+1)//2, winner)
        # Finally do the semis and championship
        self.set_predicted('W', 6, 1, self.get_winner(self.get_predicted('W', 5, 1),
                                                      self.get_predicted('X', 5, 1), predictions)) 
        self.set_predicted('Y', 6, 1, self.get_winner(self.get_predicted('Y', 5, 1), 
                                                      self.get_predicted('Z', 5, 1), predictions))
        self.set_predicted('W', 7, 1, self.get_winner(self.get_predicted('W', 6, 1),
                                                      self.get_predicted('Y', 6, 1), predictions))
        
    def propagate_loss(self, region, round, slot, loser):
        if round < 7:
            next_slot = (1+slot)//2
            next_region = self.next_region(region, round)
            next_round = round+1
            incorrect = self.get_predicted(next_region, next_round, next_slot) == loser
            if incorrect:
                self.set_correct(next_region, next_round, next_slot, False)
                self.propagate_loss(next_region, next_round, next_slot, loser)
    
    def add_winner(self, round, winner):
        b = self.bracket
        game = b.loc[(b['round'] == round) & (b['actual'] == winner)]
        slot = game['slot'].iloc[0]
        region = game['region'].iloc[0]
        next_slot = self.get_slot(slot) if round == 0 else (1+slot)//2
        next_region = self.next_region(region, round)
        self.set_actual(next_region, round+1, next_slot, winner)
        correct = self.get_correct(next_region, round+1, next_slot)
        if correct != False: # Haven't already updated
            is_correct = self.is_correct(next_region, round+1, next_slot, winner)
            self.set_correct(next_region, round+1, next_slot, is_correct)
            if not is_correct:
                loser = self.get_predicted(next_region, round+1, next_slot)
                self.propagate_loss(next_region, round+1, next_slot, loser)

    mens_rounds = {134: 0, 135: 0, 136: 1, 137: 1, 138: 2, 139: 2, 
                   143: 3, 144: 3, 145: 4, 146: 4, 152: 5, 154: 6}
    womens_rounds = {135: 0, 136: 0, 137: 1, 138: 1, 139: 2, 140: 2, 
                     144: 3, 145: 3, 146: 4, 147: 4, 151: 5, 153: 6}

    def round_to_day(self, round):
        rounds = self.mens_rounds if self.which == 'M' else self.womens_rounds
        for day in rounds:
            if rounds[day] == round: return day

    def add_results(self, num=None):
        season = self.season
        results = read_data("NCAATourneyCompactResults", self.which)
        results = results.loc[results['Season'] == season]
        results.sort_values('DayNum', inplace=True)
        for i in range(len(results) if num==None else num):
            day = results.iloc[i]['DayNum']
            round = (self.mens_rounds if self.which=='M' else self.womens_rounds)[day]
            self.add_winner(round, results.iloc[i]['WTeamID'])

    def show(self, item_width=10, use_unicode=False):
        sb = ShowBracket(self, item_width)
        sb.show(use_unicode)

    def score(self): # Current and max
        current = max = 0
        for i in range(len(self.bracket)):
            round = self.bracket.iloc[i]['round']
            correct = self.bracket.iloc[i]['correct']
            if round > 1:
                if correct == True: current += 2**(round-2)
                if correct != False: max += 2**(round-2)
            elif round == 1:
                if correct == True: current += 1
                if correct == 'waiting' or correct == True: max += 1
                # For progressive
                if self.bracket.iloc[i]['pred'] == None: max += 1
        print("Current score: %d; Max possible: %d" %(current, max))
        return current, max
    
class ProgressiveBracket(Bracket):
    predictions = None

    def fill(self):
        print("Filling not available for the progressive bracket")

    def other_slot(self, region, round, slot):
        if round < 5:
            return region, slot + (+1 if slot%2 == 1 else -1)
        elif round == 5:
            return {'W': 'X', 'X': 'W', 'Y': 'Z', 'Z': 'Y'}[region], 1
        else:
            return {'W': 'Y', 'Y': 'W'}[region], 1

    def add_winner(self, round, winner):
        if self.predictions is None:
            self.predictions = read_data("NCAATourneyPredictions", self.which)
        b = self.bracket
        game = b.loc[(b['round'] == round) & (b['actual'] == winner)]
        slot = game['slot'].iloc[0]
        region = game['region'].iloc[0]
        nround = round+1
        nregion = self.next_region(region, round)
        nslot = self.get_slot(slot) if round == 0 else (1+slot)//2
        # Find the losing team
        if round == 0:
            winner_code = game['playin'].iloc[0]
            other_code = 'b' if winner_code == 'a' else 'a'
            loser = self.get_cell(region, 0, slot)
            loser = loser.loc[loser['playin'] == other_code]['actual'].iloc[0]
        else:
            loser_region, loser_slot = self.other_slot(region, round, slot)
            loser = self.get_actual(loser_region, round, loser_slot)
        predicted = self.get_winner(winner, loser, self.predictions)
        self.set_predicted(nregion, nround, nslot, predicted)
        self.set_actual(nregion, nround, nslot, winner)
        self.set_correct(nregion, nround, nslot, winner == predicted)

if __name__ == "__main__":
    for which, label in [('M', "Men's"), ('W', "Women's")]:
        print("\n=== %s 2026 Regular Bracket ===\n" % label)
        # Ensure ML predictions exist (can take a few minutes on first run)
        pred_path = f"predictions/{which}NCAATourneyPredictions.csv"
        if not os.path.exists(pred_path):
            os.makedirs("predictions", exist_ok=True)
            try:
                from make_predictions_ml import (
                    generate_global_predictions_csv,
                    train_time_series_gb,
                )
                season = 2026
                # Train on last 5 seasons ending at 2025, weighted toward recent
                bundle = train_time_series_gb(which, train_end_season=2025, train_years=5, recency_decay=0.7)
                generate_global_predictions_csv(which, bundle, season, pred_path)
            except Exception as e:
                print(f"WARNING: could not generate ML predictions ({e}). Using seed-based fallback.")
        b = Bracket(2026, which)
        b.seed()
        b.fill()
        b.show()
