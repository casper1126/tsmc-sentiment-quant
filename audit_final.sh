echo "=== PTT SENTIMENT ===" 
head -5 data/processed/ppt_sentiment.csv
echo "..." 
tail -3 data/processed/ppt_sentiment.csv
echo ""
echo "=== NEWS SENTIMENT ==="
head -5 data/processed/news_sentiment.csv
echo "..."
tail -3 data/processed/news_sentiment.csv
echo ""
echo "=== MATRIX COVERAGE ==="
echo "Total matrix rows: $(wc -l < output/TSMC_matrix_2000-2026.csv)"
echo "Date range (from matrix):"
head -2 output/TSMC_matrix_2000-2026.csv | tail -1 | cut -d, -f1
tail -1 output/TSMC_matrix_2000-2026.csv | cut -d, -f1
echo ""
echo "=== ABLATION RESULTS ===" 
cat output/ablation_results.csv
