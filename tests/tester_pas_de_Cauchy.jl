@doc doc"""
Tester le calcul du Pas de Cauchy
"""

function tester_pas_de_Cauchy(Pas_De_Cauchy::Function)
	@testset "Le pas de Cauchy" begin
		@testset "Cas test 1" begin
			g = [0; 0]
			H = [7 0 ; 0 2]
			delta = 1
			s, e = Pas_De_Cauchy(g, H, delta)
			@testset "s" begin
				@test s == [0; 0]
			end
			@testset "e" begin
				@test e == 0
			end
		end

		@testset "Cas test 2" begin
			g = [1; 6]
			H = I
			delta = norm(g)-1
			s, e = Pas_De_Cauchy(g, H, delta)
			@testset "s" begin
				@test s == -delta/norm(g)*g
			end
			@testset "e" begin
				@test e == -1
			end
		end

		@testset "Cas test 3" begin
			g = [5; 5]
			H = I
			delta = norm(g)+1
			s, e = Pas_De_Cauchy(g, H, delta)
			@testset "s" begin
				@test s == -((transpose(g)*H*g) \ (-(-(norm(g)^2))))*g
			end
			@testset "e" begin
				@test e == 1
			end
		end

		@testset "Cas test 4" begin
			g = [4; 3]
			H = -I
			delta = norm(g)-1
			s, e = Pas_De_Cauchy(g, H, delta)
			@testset "s" begin
				@test s == -delta/norm(g)*g
			end
			@testset "e" begin
				@test e == -1
			end
		end

		@testset "Cas test 5" begin
			g = [3; 4]
			H = -I
			delta = norm(g)+1
			s, e = Pas_De_Cauchy(g, H, delta)
			@testset "s" begin
				@test s == -delta/norm(g)*g
			end
			@testset "e" begin
				@test e == -1
			end
		end
	end
end
