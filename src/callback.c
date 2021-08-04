/*
 * Function pointers for callbacks for random number generators.
 */
double (*unif_rand_ptr)(void) = 0;
double (*norm_rand_ptr)(void) = 0;
double (*exp_rand_ptr)(void) = 0;

double unif_rand(void)
{
    return (*unif_rand_ptr)();
}
double norm_rand(void)
{
    return (*norm_rand_ptr)();
}
double exp_rand(void)
{
    return (*exp_rand_ptr)();
}
