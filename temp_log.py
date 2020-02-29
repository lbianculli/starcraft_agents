# additionally, remember that vol is based on 3-yr stdev

class Accruals(CustomFactor):
    inputs = []
    window_length = 1
    
    def compute(self, today, assets, out, accruals):
        out[:] = accruals[-1]

class Profitability(CustomFactor): # gpa, roe, roa [f]cf/a, gross margin, accrual screen
    inputs = [factset.Fundamentals.gross_inc.qf,
              factset.Fundamentals.assets,
              factset.Fundamentals.roe_qf,
              factset.Fundamentals.roa_qf,
              factset.Fundamentals.free_cf_fcfe_qf,  # subbing fcf/e for fcf/a for now
              factset.Fundamentals.gross_margin_qf
             ]
    window_length = 1  # should these be quarterly? Does it matter?
    
    def compute(self, today, assets, out, gross_profit, total_assets, roe, roa, fcfe, gm):
        # should i rank in here? does it matter?
        gpa = gross_profit[-1] / total_assets[-1]
                
        out[:] = np.mean([gpa, roe, roa, fcfe, gm], axis=0)
        
        
class Growth(CustomFactor):  # inputs same as above
    inputs = [factset.Fundamentals.gross_inc.qf,
              factset.Fundamentals.assets,
              factset.Fundamentals.roe_af,
              factset.Fundamentals.roa_af,
              factset.Fundamentals.free_cf_fcfe_af,  # subbing fcf/e for fcf/a for now
              factset.Fundamentals.gross_margin_af
             ]
    window_length = 5  # double check: will this give me 5 years?
    
    def compute(self, today, assets, out, gross_profit, total_assets, roe, roa, fcfe, gm):
        gpa_growth = (gross_profit[-1] / total_assets[-1]) / (gross_profit[0] / total_assets[0]) -1
        roe_growth = roe[-1] / roe[0] - 1
        roa_growth = roa[-1] / roa[0] - 1
        fcfe_growth = fcfe[-1] / fcfe[0] - 1
        gm_growth = gm[-1] / gm[0] - 1

        out[:] = np.mean([gpa_growth, roe_growth, roa_growth, fcfe_growth, gm_growth], axis=0)
        
class Safety(CustomFactor):  # low beta (Screen?), vol, leverage, roe vol, bankruptcy risk (?)
    inputs = [EquityPricing.close.latest,
              factset.Fundamentals.roe_qf,  # qf?

             ]
    window_length = 1  # window shouldnt be 1, but how long? 1yr? 3yr? 5yr?
    
    def compute(self, today, assets, out, price, roe, ):
        
        out[:] = 
        
        
class Payout(CustomFactor):  #  just shareholder yield... w/ fasctset preferably 
    # best way to calc div yield + earnings yield? check vs. morningstar.
    inputs = []
    window_length = 
    
    def compute(self, today, assets, out, ):
        out[:] = 
        
        
        

        
