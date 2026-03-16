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
        print("\n=== %s 2025 Regular Bracket (basic seed-based model) ===\n" % label)
        # Ensure ML predictions exist (can take a few minutes on first run)
        pred_path = f"predictions/{which}NCAATourneyPredictions.csv"
        if not os.path.exists(pred_path):
            os.makedirs("predictions", exist_ok=True)
            try:
                from make_predictions_ml import (
                    generate_global_predictions_csv,
                    train_time_series_gb,
                )
                season = 2025
                bundle = train_time_series_gb(which, train_end_season=season)
                generate_global_predictions_csv(which, bundle, season, pred_path)
            except Exception as e:
                print(f"WARNING: could not generate ML predictions ({e}). Using seed-based fallback.")
        b = Bracket(2025, which)
        b.seed()
        b.fill()
        b.show()
