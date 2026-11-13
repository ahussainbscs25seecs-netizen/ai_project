# ============================================================================
# FILE: portfolio_ai.py
# Description: AI-Powered Portfolio Builder with Parallel Processing (Optimized)
# ============================================================================

import sys
import os
import concurrent.futures

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_fetcher import StockDataFetcher
from core.indicators import TechnicalIndicators
from core.rule_engine import RuleEngine


class PortfolioAI:
    """AI-powered portfolio builder that selects stocks with strong BUY signals (Parallelized)"""
    
    def __init__(self):
        self.fetcher = StockDataFetcher()
        self.engine = RuleEngine()
        
        # PSX stocks universe
        self.all_stocks = [
            "HBL", "OGDC", "PSO", "ENGRO", "MCB", "UBL", "LUCK", "FFC", "MEBL", "PPL",
            "HUBC", "MARI", "TRG", "SYS", "EFERT", "KAPCO", "NBP", "BAFL", "ABL", "SNGP",
            "POL", "DGKC", "MLCF", "PTC", "KEL", "FCCL", "HASCOL", "APL", "ICI", "DAWH",
            "CHCC", "COLG", "NML", "NESTLE", "FHAM", "PIOC", "PAEL", "BYCO", "SEARL", "SHEL"
        ]
    
    def analyze_stock(self, ticker):
        """Analyze a single stock and return buy signal data"""
        try:
            # Fetch data (1 year to ensure correct indicators calculation)
            data, error = self.fetcher.get_stock_data(ticker, "1y")
            
            if error or data is None or data.empty:
                return None
            
            # Add technical indicators
            data = TechnicalIndicators.add_all_indicators(data)
            
            # Get trading decision
            decision, confidence, signals = self.engine.analyze(data)
            
            # Get latest price and indicators
            latest_price = float(data['Close'].iloc[-1])
            rsi = float(data['RSI'].iloc[-1])
            macd = float(data['MACD'].iloc[-1])
            
            return {
                'ticker': ticker,
                'price': latest_price,
                'decision': decision,
                'confidence': confidence,
                'rsi': rsi,
                'macd': macd,
                'signals': signals
            }
            
        except Exception as e:
            # Silently fail for individual stocks to keep scanning
            return None
    
    def scan_market_for_opportunities(self, min_confidence=0):
        """Scan all stocks in parallel and filter for ANY BUY signals"""
        buy_opportunities = []
        
        # Parallel execution for speed
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ticker = {executor.submit(self.analyze_stock, ticker): ticker for ticker in self.all_stocks}
            
            for future in concurrent.futures.as_completed(future_to_ticker):
                result = future.result()
                # Only check for BUY signal, ignore confidence threshold
                if result and result['decision'] == 'BUY':
                    buy_opportunities.append(result)
        
        # Sort by confidence (highest first)
        buy_opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        
        return buy_opportunities
    
    def calculate_risk_allocation(self, risk_level, num_opportunities):
        """Calculate number of stocks and allocation strategy based on risk level"""
        if risk_level == 'conservative':
            num_stocks = min(5, num_opportunities)
            allocation_type = 'equal'
            
        elif risk_level == 'moderate':
            num_stocks = min(4, num_opportunities)
            allocation_type = 'weighted'  # Confidence-weighted
            
        else:  # aggressive
            num_stocks = min(3, num_opportunities)
            allocation_type = 'heavy'  # Top heavy
        
        # Fallback if too few opportunities found
        if num_opportunities < num_stocks:
            num_stocks = num_opportunities
            
        return num_stocks, allocation_type
    
    def allocate_budget(self, stocks, budget, allocation_type):
        """Allocate budget across selected stocks"""
        num_stocks = len(stocks)
        if num_stocks == 0:
            return []
            
        if allocation_type == 'equal':
            # Equal allocation
            weights = [1.0/num_stocks] * num_stocks

        elif allocation_type == 'weighted':
            # Confidence-weighted allocation
            total_confidence = sum(stock['confidence'] for stock in stocks)
            if total_confidence <= 0:
                weights = [1.0/num_stocks] * num_stocks  # fallback to equal
            else:
                weights = [stock['confidence'] / total_confidence for stock in stocks]

        else:  # heavy
            # Exponential weighting favoring top stocks
            confidences = [stock['confidence'] for stock in stocks]
            exp_confidences = [conf ** 2 for conf in confidences]
            total_exp = sum(exp_confidences)
            if total_exp <= 0:
                weights = [1.0/num_stocks] * num_stocks  # fallback to equal
            else:
                weights = [exp / total_exp for exp in exp_confidences]

        return weights
    
    def generate_portfolio(self, budget, risk_level='moderate'):
        """
        Generate optimized portfolio based on budget and risk level
        NOW INCLUDES ALL STOCKS WITH BUY SIGNALS!
        """
        print(f"\n{'='*60}")
        print(f"[PORTFOLIO] Generating Portfolio - Budget: PKR {budget:,.0f}")
        print(f"Risk Level: {risk_level.upper()}")
        print(f"{'='*60}\n")
        
        # Step 1: Scan market for BUY opportunities (Parallel)
        # Use min_confidence=0 to get ALL BUY signals regardless of confidence
        buy_opportunities = self.scan_market_for_opportunities(min_confidence=0)
        
        print(f"[OK] Found {len(buy_opportunities)} stocks with BUY signals (Ignoring confidence threshold)\n")
        
        if not buy_opportunities:
            return {
                'success': False,
                'message': 'No BUY opportunities found right now. Market might be bearish.',
                'stocks': [],
                'summary': {}
            }
        
        # Step 2: Select how many stocks + which allocation strategy, based on RISK LEVEL.
        # buy_opportunities is already sorted by confidence (highest first).
        num_stocks, allocation_type = self.calculate_risk_allocation(
            risk_level, len(buy_opportunities)
        )
        selected_stocks = buy_opportunities[:num_stocks]

        print(f"[INFO] Risk '{risk_level}' -> {len(selected_stocks)} stocks, "
              f"'{allocation_type}' allocation:\n")
        for stock in selected_stocks:
            print(f"   - {stock['ticker']}: {stock['confidence']}% confidence")

        # Step 3: Compute per-stock weights from the risk strategy
        #   conservative -> equal, moderate -> confidence-weighted, aggressive -> top-heavy
        weights = self.allocate_budget(selected_stocks, budget, allocation_type)

        print(f"\n[INFO] Allocation weights: "
              f"{[f'{w:.1%}' for w in weights]}\n")

        # Step 4: Calculate shares to buy per weighted allocation
        portfolio_items = []
        total_invested = 0

        for stock, weight in zip(selected_stocks, weights):
            price = stock['price']
            stock_budget = budget * weight

            # Calculate shares (whole shares only)
            shares = int(stock_budget / price) if price > 0 else 0

            if shares >= 1:  # Only include if we can buy at least 1 share
                investment = shares * price
                total_invested += investment

                portfolio_items.append({
                    'ticker': stock['ticker'],
                    'price': price,
                    'shares': shares,
                    'investment': investment,
                    'allocation_percent': 0,  # Calc later
                    'confidence': stock['confidence'],
                    'rsi': stock['rsi'],
                    'macd': stock['macd'],
                    'signals': stock['signals']
                })

                print(f"   [OK] {stock['ticker']}: {shares} shares @ PKR {price:.2f} = PKR {investment:,.2f}")

        # Recalculate percent
        for item in portfolio_items:
            item['allocation_percent'] = (item['investment'] / budget) * 100 if budget > 0 else 0

        summary = {
            'total_budget': budget,
            'total_invested': total_invested,
            'cash_remaining': budget - total_invested,
            'num_stocks': len(portfolio_items),
            'percent_invested': (total_invested / budget) * 100 if budget > 0 else 0,
            'risk_level': risk_level,
            'allocation_strategy': allocation_type
        }
        
        print(f"\n{'='*60}")
        print(f"[OK] Portfolio Generated!")
        print(f"Total Stocks: {len(portfolio_items)}")
        print(f"Total Invested: PKR {total_invested:,.2f}")
        print(f"Cash Remaining: PKR {budget - total_invested:,.2f}")
        print(f"{'='*60}\n")
        
        return {
            'success': True,
            'message': f'Successfully generated a {risk_level} portfolio with {len(portfolio_items)} stocks',
            'stocks': portfolio_items,
            'summary': summary
        }

if __name__ == '__main__':
    ai = PortfolioAI()
    print("Testing Parallel Portfolio Generation...")
    res = ai.generate_portfolio(100000, 'moderate')
    print(res)
