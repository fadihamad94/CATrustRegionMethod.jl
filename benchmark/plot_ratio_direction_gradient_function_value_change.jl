using CSV, Plots, DataFrames

const problems_paper_list =  ["ALLINITU", "ARGLINA", "BARD", "BEALE", "BIGGS6", "BOX3", "BRKMCC", "BROWNAL", "BROWNBS", "BROWNDEN", "CHNROSNB", "CLIFF", "CUBE", "DENSCHNA", "DENSCHNB", "DENSCHNC", "DENSCHND", "DENSCHNE", "DENSCHNF", "DJTL", "ENGVAL2", "ERRINROS", "EXPFIT", "GENROSEB", "GROWTHLS", "GULF", "HAIRY", "HATFLDD", "HATFLDE", "HEART6LS", "HEART8LS", "HELIX", "HIMMELBB", "HUMPS", "HYDC20LS", "JENSMP", "KOWOSB", "LOGHAIRY", "MANCINO", "MEXHAT", "MEYER3", "OSBORNEA", "OSBORNEB", "PALMER5C", "PALMER6C", "PALMER7C", "PALMER8C", "PARKCH", "PENALTY2", "PENALTY3", "PFIT1LS", "PFIT2LS", "PFIT3LS", "PFIT4LS", "ROSENBR", "S308", "SENSORS", "SINEVAL", "SISSER", "SNAIL", "STREG", "TOINTGOR", "TOINTPSP", "VARDIM", "VIBRBEAM", "WATSON", "YFITU"]

function readDataFrames(directoryName::String, cutestProblems::Vector{String})
	all_dataframes = Dict()
	for cutestProblem in cutestProblems
		filePath = string(directoryName, "/", cutestProblem, ".csv")
		df = DataFrame(CSV.File(filePath))
		all_dataframes[cutestProblems] = df
	end
	return all_dataframes
end

function plot(directoryName::String, all_dataframes::Dict)
	plotName = "ratio_direction_next_gradient_function_value_change.png"
	plotPath = string(directoryName, "/", plotName)
	for (key, value) in all_dataframes
		stepTypeMarker = []
		for stepType in value.stepType
			if stepType == "Newton"
				push!(stepTypeMarker, :N)
			else
				push!(stepTypeMarker, :T)
			end
		end
		plot!(value.k,
	        value.ratio,
	        label=key,
	        ylabel="Ratio",
	        xlabel="Total number of iterations",
	        # xlims=(1, max_it),
	        legend=:topright,
			markershape=stepTypeMarker
	        #xaxis=:log10,
	        #yaxis=:log10,
	    )
	end
	png(plotPath)
end

function plotResults()
	directoryName = "/afs/pitt.edu/home/f/a/fah33/CAT/scripts/benchmark/results/CAT_GALAHAD_FACTORIZATION"
	all_dataframes = readDataFrames(directoryName, problems_paper_list)
	plot(directoryName, all_dataframes)
end

function plotNearConvexityRatio()
	optimization_method = "CAT_GALAHAD_FACTORIZATION"
	directoryName = "/afs/pitt.edu/home/f/a/fah33/CAT/scripts/benchmark/results/$optimization_method"
	filePath = string(directoryName, "/", "table_near_convexity_ratio_$optimization_method.csv")
	df = DataFrame(CSV.File(filePath))
	plot!(df.k,
		df.ratio,
		ylabel="Ratio",
		xlabel="Cutest problem"
	)
	filePath = string(directoryName, "/", "near_convexity_ratio_$optimization_method.png")
	png(filePath)
end

plotResults()
